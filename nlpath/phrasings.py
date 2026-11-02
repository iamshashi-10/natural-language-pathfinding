"""Instruction wordings for the benchmark (plan §4.1).

Slots: {G} goal, {X} {X2} forbidden, {W} waypoint, {A} {B} two ordered
waypoints, {C} the preferred corridor, {Y} an avoided room. Region names are
written in plain English ("store room"); the parser sees the legend
(``store_room``) and must map one to the other.

* ``TEMPLATES``: several wordings per category. A templated instruction uses
  three different wordings of the same intent (original + two paraphrases).
* ``HANDWRITTEN``: 30 deliberately natural, messy patterns, each with its own
  three wordings and an intent shape. Some combine constraints, use arrows or
  reversed order, or say "avoid ... completely" (hard) vs "if you can" (soft).
"""

from __future__ import annotations

TEMPLATES: dict[str, list[str]] = {
    "plain": [
        "Go to the {G}.",
        "Take me to the {G}.",
        "Head over to the {G}, please.",
        "I need to get to the {G}.",
        "Navigate to the {G}.",
    ],
    "prohibition": [
        "Reach the {G} without entering the {X}.",
        "Go to the {G}, but don't go through the {X}.",
        "Get to the {G} and stay out of the {X}.",
        "Head to the {G}; the {X} is off limits.",
        "Navigate to the {G} without ever setting foot in the {X}.",
    ],
    "stop": [
        "Stop at the {W}, then go to the {G}.",
        "Go to the {G}, but visit the {W} on the way.",
        "First pass by the {W}, and after that head to the {G}.",
        "Swing by the {W} before ending up in the {G}.",
        "I want to end in the {G}; make a stop at the {W} first.",
    ],
    "ordering": [
        "Visit the {A} before the {B}, then go to the {G}.",
        "Go to the {A}, then the {B}, and finish at the {G}.",
        "First the {A}, next the {B}, and finally the {G}.",
        "Stop by the {A} and afterwards the {B} on your way to the {G}.",
        "Head to the {G}, but check the {A} first and then the {B}.",
    ],
    "soft_prefer": [
        "Go to the {G}, and stay in the {C} if you can.",
        "Head to the {G}, preferably along the {C}.",
        "Get to the {G}; try to use the {C} where possible.",
        "Make your way to the {G}, sticking to the {C} when it's convenient.",
    ],
    "soft_avoid": [
        "Go to the {G}, and try to avoid the {Y} if possible.",
        "Head to the {G}; going through the {Y} is okay, but better not.",
        "Get to the {G}, keeping out of the {Y} unless it saves a lot.",
        "Reach the {G}, preferably not passing through the {Y}.",
    ],
}

# (category, intent shape, three wordings). Shape letters: G goal, X/X2 forbidden,
# W one waypoint, A,B two waypoints, C preferred corridor, Y avoided.
HANDWRITTEN: list[tuple[str, str, list[str]]] = [
    # --- plain
    ("plain", "G", ["ok so i need to get over to the {G} asap",
                    "can u take me to the {G}",
                    "{G}. that's where I'm going."]),
    ("plain", "G", ["Where's the {G}? Bring me there.",
                    "I've got a meeting in the {G}, lead the way",
                    "need to be in the {G} in five, go"]),
    ("plain", "G", ["just get me to the {G} however you want",
                    "{G} please, any route is fine",
                    "any way you like, just end up at the {G}"]),
    ("plain", "G", ["Could you guide me to the {G} when you get a chance?",
                    "guide me to the {G} thx",
                    "the {G} - that's the destination"]),
    ("plain", "G", ["head for the {G} and stop there",
                    "destination: {G}",
                    "bring me to the {G} and wait"]),
    # --- prohibition
    ("prohibition", "G X", ["{G} pls. and DON'T go thru the {X}, it's being cleaned",
                            "go to the {G}, {X} is closed today so skip it",
                            "to the {G} - the {X} is closed, do not enter it"]),
    ("prohibition", "G X", ["I want to reach the {G} but the {X} is a no-go zone",
                            "{G}, and whatever you do, not through the {X}",
                            "avoid the {X} completely, just get to the {G}"]),
    ("prohibition", "G X", ["never enter the {X}. the goal is the {G}",
                            "the {X} must not be entered on the way to the {G}",
                            "{G} is the target; the {X} is strictly off limits"]),
    ("prohibition", "G X", ["take me to the {G}, keep clear of the {X} at all costs",
                            "get to the {G} and don't even step into the {X}",
                            "the {X} is dangerous right now - route to the {G} around it"]),
    ("prohibition", "G X X2", ["get to the {G}, but stay out of the {X} and the {X2}",
                               "{G}; neither the {X} nor the {X2} may be entered",
                               "route to the {G} avoiding the {X} and {X2} entirely, both are locked"]),
    # --- stop
    ("stop", "G W", ["grab something from the {W} and then bring it to the {G}",
                     "pick up a box at the {W}, deliver it to the {G}",
                     "{W} first (pickup), then drop off at the {G}"]),
    ("stop", "G W", ["on the way to the {G} could you pop into the {W}?",
                     "pop into the {W} before you get to the {G}",
                     "need a quick stop at the {W}, final destination {G}"]),
    ("stop", "G W", ["go to the {G} but hit the {W} first",
                     "hit the {W}, then {G}",
                     "{W} then {G}, in that order"]),
    ("stop", "G W X", ["drop by the {W} then go to the {G}, don't cut through the {X}",
                       "visit the {W} and then the {G}; the {X} is closed",
                       "first {W}, then {G}, and keep out of the {X}"]),
    ("stop", "G W C", ["stop at the {W} and then go to the {G}, use the {C} if you can",
                       "{W} then {G}, preferably via the {C}",
                       "make a stop at the {W} on the way to the {G}, sticking to the {C} where it's easy"]),
    # --- ordering
    ("ordering", "G A B", ["do the {A}, then the {B}, and end at the {G}",
                           "{A} -> {B} -> {G}",
                           "{A} first, {B} second, {G} last"]),
    ("ordering", "G A B", ["I have errands: {A}, then {B}. After both, go to the {G}",
                           "run errands at the {A} and then the {B}, finish in the {G}",
                           "after the {A} and then the {B}, end up at the {G}"]),
    ("ordering", "G A B", ["go to the {G}, but before that the {B}, and before that the {A}",
                           "end at the {G}; before the {G} visit the {B}; before the {B} visit the {A}",
                           "the {G} is last, preceded by the {B}, which comes after the {A}"]),
    ("ordering", "G A B X", ["{A} then {B} then {G}, and never through the {X}",
                             "visit {A}, then {B}, finish at {G}; {X} is off limits",
                             "go {A}, {B}, {G} in order without entering the {X}"]),
    ("ordering", "G A B", ["make sure you see the {A} before the {B}, and wrap up in the {G}",
                           "the {B} only after the {A}; final stop the {G}",
                           "{A} has to come before {B}; end at the {G}"]),
    # --- soft
    ("soft", "G C", ["go to the {G}, try to stay in the {C} if possible",
                     "{G} please, the {C} is nicer so use it when you can",
                     "get to the {G}, mostly using the {C} if that's not a big detour"]),
    ("soft", "G Y", ["head to the {G}. the {Y} is a bit busy so maybe avoid it if you can",
                     "{G}, ideally not through the {Y} but it's fine if needed",
                     "try not to go through the {Y} on the way to the {G}, unless it's much shorter"]),
    ("soft", "G C Y", ["to the {G}, prefer the {C}, avoid the {Y} if you can",
                       "{G}; use the {C} where possible and stay away from the {Y} when it's easy",
                       "go to the {G}, the {C} is preferred and the {Y} is better avoided"]),
    ("soft", "G C X", ["go to the {G} using the {C} when you can, and don't enter the {X} at all",
                       "{G}, preferably along the {C}; the {X} is strictly closed",
                       "reach the {G} - {C} if convenient, {X} never"]),
    ("soft", "G Y", ["uh, the {G}. avoid the {Y} if it's not too much trouble",
                     "go to the {G} and if you can, skip the {Y}",
                     "{G}, and the {Y} only if there's no reasonable way around"]),
    # --- infeasible (labelled infeasible by the reference solver)
    ("infeasible", "G X", ["get to the {G} but stay out of the {X}",
                           "{G}, without entering the {X} at any point",
                           "go to the {G}; the {X} is locked"]),
    ("infeasible", "G X", ["the {X} is flooded, don't go in. I need the {G}",
                           "{G} please - {X} is a no-go",
                           "reach the {G}; under no circumstances enter the {X}"]),
    ("infeasible", "G X X2", ["{G}, but the {X} and the {X2} are both closed",
                              "go to the {G} without passing through the {X} or the {X2}",
                              "avoid {X} and {X2} entirely and get to the {G}"]),
    ("infeasible", "G W X", ["stop at the {W}, then the {G}, never through the {X}",
                             "{W} then {G}; the {X} must not be entered",
                             "visit the {W} on the way to the {G} and stay out of the {X}"]),
    ("infeasible", "G X", ["could you take me to the {G}? the {X} is off limits today",
                           "{X} is closed. destination {G}",
                           "{G}, but skip the {X} completely"]),
]
