#!/usr/bin/env python
"""
OpenAI Usage Dashboard
Displays a beautifully formatted summary of OpenAI API usage including models, tokens, and costs.

Note: This script requires an ADMIN API key from OpenAI, not a regular API key.
Get your admin key from: https://platform.openai.com/settings/organization/admin-keys
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

try:
    import requests
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    from rich.progress import track
    from rich.columns import Columns
except ImportError as e:
    print(f"Missing required packages. Install with:")
    print("pip install requests rich")
    sys.exit(1)


class OpenAIUsageTracker:
    def __init__(self):
        # Try both environment variable names
        self.api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_ADMIN_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY environment variable.\n"
                "Note: This script requires an ADMIN API key from:\n"
                "https://platform.openai.com/settings/organization/admin-keys"
            )

        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.console = Console()

        # OpenAI pricing (updated for 2025)
        self.pricing = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01},
            "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
            "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
            "o1-preview": {"input": 0.015, "output": 0.06},
            "o1-mini": {"input": 0.003, "output": 0.012},
            "o1": {"input": 0.015, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "text-embedding-3-large": {"input": 0.00013, "output": 0},
            "text-embedding-3-small": {"input": 0.00002, "output": 0},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0},
            "whisper-1": {"input": 0.006, "output": 0},  # per minute
            "tts-1": {"input": 0.015, "output": 0},       # per 1K characters
            "tts-1-hd": {"input": 0.03, "output": 0},     # per 1K characters
            "dall-e-2": {"input": 0.02, "output": 0},     # per image
            "dall-e-3": {"input": 0.04, "output": 0},     # per image (standard)
        }

    def get_usage_data(self, days: int = 30) -> Dict:
        """Fetch usage data from OpenAI Completions Usage API"""
        start_time = int(time.time()) - (days * 24 * 60 * 60)

        url = f"{self.base_url}/organization/usage/completions"
        params = {
            "start_time": start_time,
            "bucket_width": "1d",
            "group_by": ["model"],  # Group by model to get per-model breakdown
            "limit": days
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                self.console.print("[red]Authentication failed. Make sure you're using an ADMIN API key.[/red]")
                self.console.print("[yellow]Get your admin key from: https://platform.openai.com/settings/organization/admin-keys[/yellow]")
            elif response.status_code == 403:
                self.console.print("[red]Access forbidden. This endpoint requires an ADMIN API key.[/red]")
                self.console.print("[yellow]Regular API keys don't work for usage endpoints.[/yellow]")
            else:
                self.console.print(f"[red]HTTP Error {response.status_code}: {e}[/red]")
            return {}
        except requests.exceptions.RequestException as e:
            self.console.print(f"[red]Error fetching usage data: {e}[/red]")
            return {}

    def get_costs_data(self, days: int = 30) -> Dict:
        """Fetch cost data from OpenAI Costs API"""
        start_time = int(time.time()) - (days * 24 * 60 * 60)

        url = f"{self.base_url}/organization/costs"
        params = {
            "start_time": start_time,
            "bucket_width": "1d",
            "limit": days
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.console.print(f"[yellow]Warning: Could not fetch cost data: {e}[/yellow]")
            return {}

    def process_usage_data(self, usage_data: Dict) -> Tuple[Dict, int, int]:
        """Process usage data and calculate totals"""
        model_usage = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_requests = 0

        if 'data' not in usage_data:
            return {}, 0, 0

        for bucket in usage_data['data']:
            for result in bucket.get('results', []):
                model = result.get('model', 'unknown')
                input_tokens = result.get('input_tokens', 0)
                output_tokens = result.get('output_tokens', 0)
                requests_count = result.get('num_model_requests', 0)

                if model not in model_usage:
                    model_usage[model] = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0,
                        'requests': 0,
                        'estimated_cost': 0.0
                    }

                model_usage[model]['input_tokens'] += input_tokens
                model_usage[model]['output_tokens'] += output_tokens
                model_usage[model]['total_tokens'] += input_tokens + output_tokens
                model_usage[model]['requests'] += requests_count

                # Calculate estimated cost
                model_key = model.lower() if model else 'unknown'
                cost = 0.0

                # Find matching pricing model
                for pricing_model in self.pricing:
                    if pricing_model in model_key or model_key in pricing_model:
                        input_cost = (input_tokens / 1000) * self.pricing[pricing_model]["input"]
                        output_cost = (output_tokens / 1000) * self.pricing[pricing_model]["output"]
                        cost = input_cost + output_cost
                        break

                model_usage[model]['estimated_cost'] += cost

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_requests += requests_count

        return model_usage, total_input_tokens + total_output_tokens, total_requests

    def process_costs_data(self, costs_data: Dict) -> float:
        """Process cost data to get actual total cost"""
        total_cost = 0.0

        if 'data' not in costs_data:
            return 0.0

        for bucket in costs_data['data']:
            for result in bucket.get('results', []):
                amount = result.get('amount', {})
                if isinstance(amount, dict):
                    total_cost += amount.get('value', 0.0)
                else:
                    total_cost += float(amount) if amount else 0.0

        return total_cost

    def create_usage_table(self, model_usage: Dict, total_tokens: int) -> Table:
        """Create a rich table for usage data"""
        table = Table(title="📊 Model Usage Breakdown", box=box.ROUNDED)

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Requests", justify="right", style="blue")
        table.add_column("Input Tokens", justify="right", style="green")
        table.add_column("Output Tokens", justify="right", style="yellow")
        table.add_column("Total Tokens", justify="right", style="magenta")
        table.add_column("Est. Cost", justify="right", style="red")
        table.add_column("Usage %", justify="right", style="white")

        # Sort by total tokens (descending)
        sorted_models = sorted(model_usage.items(), key=lambda x: x[1]['total_tokens'], reverse=True)

        for model, data in sorted_models:
            if data['total_tokens'] == 0:
                continue

            requests = f"{data['requests']:,}"
            input_tokens = f"{data['input_tokens']:,}"
            output_tokens = f"{data['output_tokens']:,}"
            total_model_tokens = f"{data['total_tokens']:,}"
            cost = f"${data['estimated_cost']:.4f}"
            percentage = f"{(data['total_tokens'] / total_tokens * 100):.1f}%" if total_tokens > 0 else "0%"

            # Truncate long model names
            display_model = model if len(model) <= 25 else model[:22] + "..."

            table.add_row(
                display_model, requests, input_tokens, output_tokens,
                total_model_tokens, cost, percentage
            )

        return table

    def create_summary_panels(self, total_tokens: int, total_requests: int,
                            estimated_cost: float, actual_cost: float, days: int) -> List[Panel]:
        """Create summary panels with key metrics"""
        panels = []

        # Usage Summary Panel
        usage_text = Text()
        usage_text.append("🚀 ", style="bold blue")
        usage_text.append(f"Total Tokens: ", style="bold")
        usage_text.append(f"{total_tokens:,}\n", style="cyan")

        usage_text.append("📡 ", style="bold green")
        usage_text.append(f"Total Requests: ", style="bold")
        usage_text.append(f"{total_requests:,}\n", style="green")

        usage_text.append("📅 ", style="bold yellow")
        usage_text.append(f"Period: ", style="bold")
        usage_text.append(f"Last {days} days", style="yellow")

        panels.append(Panel(usage_text, title="📈 Usage Summary", border_style="blue"))

        # Cost Summary Panel
        cost_text = Text()
        if actual_cost > 0:
            cost_text.append("💰 ", style="bold green")
            cost_text.append(f"Actual Cost: ", style="bold")
            cost_text.append(f"${actual_cost:.4f}\n", style="green")

        cost_text.append("🧮 ", style="bold magenta")
        cost_text.append(f"Estimated Cost: ", style="bold")
        cost_text.append(f"${estimated_cost:.4f}\n", style="magenta")

        avg_daily_cost = (actual_cost if actual_cost > 0 else estimated_cost) / days if days > 0 else 0
        cost_text.append("📊 ", style="bold cyan")
        cost_text.append(f"Avg Daily: ", style="bold")
        cost_text.append(f"${avg_daily_cost:.4f}", style="cyan")

        panels.append(Panel(cost_text, title="💸 Cost Summary", border_style="green"))

        return panels

    def display_dashboard(self, days: int = 30):
        """Display the complete usage dashboard"""
        with self.console.status("[bold green]Fetching OpenAI usage data..."):
            usage_data = self.get_usage_data(days)
            costs_data = self.get_costs_data(days)

        if not usage_data:
            self.console.print("[red]No usage data available or API error[/red]")
            return

        # Process data
        model_usage, total_tokens, total_requests = self.process_usage_data(usage_data)
        actual_cost = self.process_costs_data(costs_data)
        estimated_cost = sum(data['estimated_cost'] for data in model_usage.values())

        # Clear screen and display header
        self.console.clear()
        self.console.print("\n")
        self.console.print("🤖 [bold blue]OpenAI Usage Dashboard[/bold blue] 🤖", justify="center")
        self.console.print("=" * 60, style="blue", justify="center")
        self.console.print("\n")

        # Display summary panels side by side
        summary_panels = self.create_summary_panels(
            total_tokens, total_requests, estimated_cost, actual_cost, days
        )
        self.console.print(Columns(summary_panels, equal=True))
        self.console.print("\n")

        # Display usage table
        if model_usage:
            usage_table = self.create_usage_table(model_usage, total_tokens)
            self.console.print(usage_table)
        else:
            self.console.print("[yellow]No detailed usage data available[/yellow]")

        # Display additional info
        self.console.print("\n")
        if actual_cost > 0:
            cost_diff = abs(actual_cost - estimated_cost)
            if cost_diff > 0.01:  # Only show if difference is significant
                diff_pct = (cost_diff / max(actual_cost, estimated_cost)) * 100
                self.console.print(f"💡 [italic]Cost difference: ±${cost_diff:.4f} ({diff_pct:.1f}%)[/italic]", style="dim")

        # Footer
        self.console.print("📝 [italic]Costs are estimates based on current pricing[/italic]", style="dim")
        self.console.print(f"🕐 [italic]Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/italic]", style="dim")


def main():
    """Main function to run the usage dashboard"""
    try:
        tracker = OpenAIUsageTracker()

        # Allow custom date range via command line argument
        days = 30
        if len(sys.argv) > 1:
            try:
                days = int(sys.argv[1])
                if days <= 0 or days > 365:
                    raise ValueError()
            except ValueError:
                print("Usage: python openai_usage.py [days]")
                print("Days must be a positive integer (1-365)")
                sys.exit(1)

        tracker.display_dashboard(days)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDashboard interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
