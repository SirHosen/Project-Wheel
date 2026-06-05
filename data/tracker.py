import os
import json
import pandas as pd
from datetime import datetime

class Tracker:
    """Unified Data Pipeline and History Tracker."""
    
    def __init__(self, history_file="data/history.json"):
        self.history_file = history_file
        self.ensure_directory()
        self.data = self.load_data()
        
    def ensure_directory(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
    def load_data(self) -> dict:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                import shutil
                import logging
                corrupt_file = self.history_file.replace(".json", ".corrupt.json")
                try:
                    shutil.copy2(self.history_file, corrupt_file)
                    logging.error(f"Failed to load {self.history_file}. Backed up to {corrupt_file}")
                except Exception:
                    pass
                
        return {
            "current_capital": 1000,
            "total_predictions": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0,
            "history": [] # list of dicts: {timestamp, actual_number, predicted_number, profit_change}
        }
        
    def save_data(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Error saving data: {e}")
            
    def record_result(self, actual_number: int, predicted_number: int, profit_change: int):
        is_win = (actual_number == predicted_number) if predicted_number else False
        
        self.data["total_predictions"] += 1
        if is_win:
            self.data["wins"] += 1
        else:
            self.data["losses"] += 1
            
        self.data["profit"] += profit_change
        self.data["current_capital"] += profit_change
        
        self.data["history"].append({
            "timestamp": datetime.now().isoformat(),
            "actual_number": actual_number,
            "predicted_number": predicted_number,
            "profit_change": profit_change,
            "is_win": is_win
        })
        
        self.save_data()
        
    def get_recent_actuals(self, limit=100) -> list:
        """Returns a list of the recent actual numbers for prediction models."""
        history = self.data.get("history", [])
        return [record["actual_number"] for record in history][-limit:]
        
    def get_stats(self) -> dict:
        total = self.data["total_predictions"]
        win_rate = (self.data["wins"] / total * 100) if total > 0 else 0.0
        return {
            "capital": self.data["current_capital"],
            "total": total,
            "wins": self.data["wins"],
            "losses": self.data["losses"],
            "profit": self.data["profit"],
            "win_rate": win_rate
        }
        
    def export_csv(self, output_path="data/history.csv"):
        """Exports history to CSV using Pandas for analytics."""
        df = pd.DataFrame(self.data["history"])
        df.to_csv(output_path, index=False)
        return output_path

    def get_streak(self) -> int:
        """Returns the current streak of wins."""
        history = self.data.get("history", [])
        streak = 0
        for record in reversed(history):
            if record.get("is_win", False):
                streak += 1
            else:
                break
        return streak
        
    def reset_data(self):
        """Resets all data to initial state."""
        self.data = {
            "current_capital": 1000,
            "total_predictions": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0,
            "history": []
        }
        self.save_data()

