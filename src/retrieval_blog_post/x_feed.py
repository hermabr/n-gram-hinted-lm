"""The other posts of twitter.py's mock home timeline, anonymized: placeholder authors and lorem ipsum text, no
pictures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Author:
    name: str
    handle: str
    avatar: str | None = None  # its 96 px picture, in assets/twitter/; None draws a blank circle
    verified: bool = False


@dataclass(frozen=True)
class Post:
    author: Author
    ago: str  # how long ago it was posted, as the timeline shows it
    text: str  # with its links expanded (twitter.py shortens them as X does)
    counts: tuple[int, int, int, int]  # replies, reposts, likes, views


# The post above the thread.
ABOVE = Post(
    Author("Lorem Ipsum", "loremipsum"),
    "2h",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
    "magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation.\n\n"
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore:",
    (119, 248, 2595, 154660),
)
# The posts below it.
BELOW = (
    Post(
        Author("Dolor Sit", "dolorsit"),
        "6h",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem "
        "aperiam.\n\nNemo enim ipsam voluptatem quia voluptas sit aspernatur.",
        (27, 126, 1059, 43090),
    ),
    Post(
        Author("Amet Consectetur", "ametconsectetur"),
        "3h",
        "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet?",
        (45, 11, 1219, 174241),
    ),
    Post(
        Author("Adipiscing", "adipiscing_"),
        "6h",
        "ut enim ad minima veniam.\n\nquis autem vel eum iure.",
        (16, 55, 1156, 160267),
    ),
    Post(
        Author("Elit Sed", "elitsed"),
        "15h",
        "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum.\n\n"
        "Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo minus id quod maxime.",
        (152, 56, 1131, 54084),
    ),
)
