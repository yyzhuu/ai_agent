from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.style import Style


console = Console()
blue_border_style = Style(color="bright_blue")
red_border_style = Style(color="red")
green_border_style = Style(color="bright_green")


def log(content: str) -> None:
    console.log(content)


def log_panel(
    title: str,
    content: str,
    border_style: Style = blue_border_style,
) -> None:
    console.log(Panel(content, title=title, border_style=border_style))
