"""
CSV Generator: converts Event list → competition submission CSV.

Output format per row:
    품목명, 이벤트 번호, 구매/반환 여부, 이벤트 후 재고 수량, 총액

Example:
    a1_steak_sauce, Event 7, 구매, 재고:2, 총액:$23.5
"""

import csv
import os
from typing import List, Dict
from event_detector import Event


def load_prices(prices_csv: str) -> Dict[int, tuple]:
    """
    prices.csv format:  class_id, class_name, price_usd
    Returns {class_id: (class_name, price_usd)}
    """
    prices = {}
    with open(prices_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_id = int(row["class_id"])
            prices[cls_id] = (row["class_name"], float(row["price_usd"]))
    return prices


def events_to_csv(
    events: List[Event],
    prices: Dict[int, tuple],
    out_path: str,
    initial_inventory: Dict[int, int],
):
    """
    Generates the submission CSV.

    initial_inventory: {class_id: initial_stock} before any events
    """
    # Running inventory to compute 총액 (sum of all purchases so far)
    running_total: Dict[int, float] = {}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["품목명", "이벤트 번호", "구매/반환 여부",
                         "이벤트 후 재고 수량", "총액"])

        for event in events:
            cls_id = event.class_id
            if cls_id not in prices:
                continue  # skip unknown classes

            name, price = prices[cls_id]

            # Accumulate total purchase amount per class
            if cls_id not in running_total:
                running_total[cls_id] = 0.0

            if event.action == "구매":
                purchased = event.before - event.after
                running_total[cls_id] += price * purchased
            else:  # 반환
                returned = event.after - event.before
                running_total[cls_id] -= price * returned
                running_total[cls_id] = max(0.0, running_total[cls_id])

            total = running_total[cls_id]
            writer.writerow([
                name,
                f"Event {event.event_num}",
                event.action,
                f"재고:{event.after}",
                f"총액:${total:.1f}",
            ])

    print(f"Saved {len(events)} events → {out_path}")
