import functools
import os

from rich.console import Console

console = Console()
CITATION_LINK = "https://github.com/BrainLesion/PeTu#citation"


def citation_reminder(func):
    """
    Decorator to remind users to cite the PeTu manuscript

    The reminder is shown when the environment variable
    `PETU_CITATION_REMINDER` is set to "true" (default).
    To disable the reminder, set the environment variable to "false".

    Environment variable used:
    - PETU_CITATION_REMINDER: Controls whether the reminder is shown.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("PETU_CITATION_REMINDER", "true").lower() == "true":
            console.rule("Thank you for using [bold]PeTu[/bold]")
            console.print(
                "Please support our development by citing",
                justify="center",
            )
            console.print(
                f"{CITATION_LINK} -- Thank you!",
                justify="center",
            )
            console.rule()
            console.line()
        return func(*args, **kwargs)

    return wrapper
