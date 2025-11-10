"""Central ALFRED action catalog for EmbodiedBench planner post-processing.

IMPORTANT SCOPE NOTE
--------------------
This catalog enumerates the static, globally referenced ALFRED high-level
action space (object-centric: find / pick up / open / close / slice, etc.).
It is used by the plan validation / repair logic ONLY for the `eb-alf`
environment. The EB-Habitat environment (`eb-hab`) exposes a *dynamic* per-
episode action list derived from simulator task definitions (navigation,
pick, place, open, close with instance-specific natural language). Habitat
actions therefore MUST NOT be canonicalized through this static mapping.

The validator now gates repairs:
    * eb-alf : full catalog-based semantic repair (name-first precedence)
    * eb-hab : lightweight field normalization only (no catalog usage)

Having a single authoritative source for ALFRED reduces drift between prompt
enumerations and runtime validation/repair. Extended tail (208-217) retains
duplicate semantic entries intentionally for backwards compatibility with
alternate enumerations.
"""
from __future__ import annotations

from typing import Dict

# Core (0-207) plus extended (208-217) enumeration.
# NOTE: The wording matches the prompt enumeration EXACTLY to guarantee
# deterministic validation.
_ACTION_PAIRS = [
    (0, "find a Cart"), (1, "find a Potato"), (2, "find a Faucet"), (3, "find a Ottoman"),
    (4, "find a CoffeeMachine"), (5, "find a Candle"), (6, "find a CD"), (7, "find a Pan"),
    (8, "find a Watch"), (9, "find a HandTowel"), (10, "find a SprayBottle"), (11, "find a BaseballBat"),
    (12, "find a CellPhone"), (13, "find a Kettle"), (14, "find a Mug"), (15, "find a StoveBurner"),
    (16, "find a Bowl"), (17, "find a Toilet"), (18, "find a DiningTable"), (19, "find a Spoon"),
    (20, "find a TissueBox"), (21, "find a Shelf"), (22, "find a Apple"), (23, "find a TennisRacket"),
    (24, "find a SoapBar"), (25, "find a Cloth"), (26, "find a Plunger"), (27, "find a FloorLamp"),
    (28, "find a ToiletPaperHanger"), (29, "find a CoffeeTable"), (30, "find a Spatula"), (31, "find a Plate"),
    (32, "find a Bed"), (33, "find a Glassbottle"), (34, "find a Knife"), (35, "find a Tomato"),
    (36, "find a ButterKnife"), (37, "find a Dresser"), (38, "find a Microwave"), (39, "find a CounterTop"),
    (40, "find a GarbageCan"), (41, "find a WateringCan"), (42, "find a Vase"), (43, "find a ArmChair"),
    (44, "find a Safe"), (45, "find a KeyChain"), (46, "find a Pot"), (47, "find a Pen"),
    (48, "find a Cabinet"), (49, "find a Desk"), (50, "find a Newspaper"), (51, "find a Drawer"),
    (52, "find a Sofa"), (53, "find a Bread"), (54, "find a Book"), (55, "find a Lettuce"),
    (56, "find a CreditCard"), (57, "find a AlarmClock"), (58, "find a ToiletPaper"), (59, "find a SideTable"),
    (60, "find a Fork"), (61, "find a Box"), (62, "find a Egg"), (63, "find a DeskLamp"),
    (64, "find a Ladle"), (65, "find a WineBottle"), (66, "find a Pencil"), (67, "find a Laptop"),
    (68, "find a RemoteControl"), (69, "find a BasketBall"), (70, "find a DishSponge"), (71, "find a Cup"),
    (72, "find a SaltShaker"), (73, "find a PepperShaker"), (74, "find a Pillow"), (75, "find a Bathtub"),
    (76, "find a SoapBottle"), (77, "find a Statue"), (78, "find a Fridge"), (79, "find a Sink"),
    (80, "pick up the KeyChain"), (81, "pick up the Potato"), (82, "pick up the Pot"), (83, "pick up the Pen"),
    (84, "pick up the Candle"), (85, "pick up the CD"), (86, "pick up the Pan"), (87, "pick up the Watch"),
    (88, "pick up the Newspaper"), (89, "pick up the HandTowel"), (90, "pick up the SprayBottle"), (91, "pick up the BaseballBat"),
    (92, "pick up the Bread"), (93, "pick up the CellPhone"), (94, "pick up the Book"), (95, "pick up the Lettuce"),
    (96, "pick up the CreditCard"), (97, "pick up the Mug"), (98, "pick up the AlarmClock"), (99, "pick up the Kettle"),
    (100, "pick up the ToiletPaper"), (101, "pick up the Bowl"), (102, "pick up the Fork"), (103, "pick up the Box"),
    (104, "pick up the Egg"), (105, "pick up the Spoon"), (106, "pick up the TissueBox"), (107, "pick up the Apple"),
    (108, "pick up the TennisRacket"), (109, "pick up the Ladle"), (110, "pick up the WineBottle"), (111, "pick up the Cloth"),
    (112, "pick up the Plunger"), (113, "pick up the SoapBar"), (114, "pick up the Pencil"), (115, "pick up the Laptop"),
    (116, "pick up the RemoteControl"), (117, "pick up the BasketBall"), (118, "pick up the DishSponge"), (119, "pick up the Cup"),
    (120, "pick up the Spatula"), (121, "pick up the SaltShaker"), (122, "pick up the Plate"), (123, "pick up the PepperShaker"),
    (124, "pick up the Pillow"), (125, "pick up the Glassbottle"), (126, "pick up the SoapBottle"), (127, "pick up the Knife"),
    (128, "pick up the Statue"), (129, "pick up the Tomato"), (130, "pick up the ButterKnife"), (131, "pick up the WateringCan"),
    (132, "pick up the Vase"), (133, "put down the object in hand"), (134, "drop the object in hand"), (135, "open the Safe"),
    (136, "close the Safe"), (137, "open the Laptop"), (138, "close the Laptop"), (139, "open the Fridge"),
    (140, "close the Fridge"), (141, "open the Box"), (142, "close the Box"), (143, "open the Microwave"),
    (144, "close the Microwave"), (145, "open the Cabinet"), (146, "close the Cabinet"), (147, "open the Drawer"),
    (148, "close the Drawer"), (149, "turn on the Microwave"), (150, "turn off the Microwave"), (151, "turn on the DeskLamp"),
    (152, "turn off the DeskLamp"), (153, "turn on the FloorLamp"), (154, "turn off the FloorLamp"), (155, "turn on the Faucet"),
    (156, "turn off the Faucet"), (157, "slice the Potato"), (158, "slice the Lettuce"), (159, "slice the Tomato"),
    (160, "slice the Apple"), (161, "slice the Bread"), (162, "find a Bowl_2"), (163, "find a Cabinet_2"),
    (164, "find a CounterTop_2"), (165, "find a CounterTop_3"), (166, "find a Cup_2"), (167, "find a Drawer_2"),
    (168, "find a Drawer_3"), (169, "find a Drawer_4"), (170, "find a Drawer_5"), (171, "find a Drawer_6"),
    (172, "find a Mug_2"), (173, "find a Mug_3"), (174, "find a Plate_2"), (175, "find a Plate_3"),
    (176, "find a Pot_2"), (177, "find a StoveBurner_2"), (178, "find a StoveBurner_3"), (179, "find a StoveBurner_4"),
    (180, "open the Cabinet_2"), (181, "close the Cabinet_2"), (182, "open the Drawer_2"), (183, "close the Drawer_2"),
    (184, "open the Drawer_3"), (185, "close the Drawer_3"), (186, "open the Drawer_4"), (187, "close the Drawer_4"),
    (188, "open the Drawer_5"), (189, "close the Drawer_5"), (190, "open the Drawer_6"), (191, "close the Drawer_6"),
    (192, "find a Apple_2"), (193, "find a Apple_3"), (194, "find a Bread_2"), (195, "find a Egg_2"),
    (196, "find a Egg_3"), (197, "find a Fork_2"), (198, "find a Fork_3"), (199, "find a Knife_2"),
    (200, "find a Ladle_2"), (201, "find a PepperShaker_2"), (202, "find a PepperShaker_3"), (203, "find a Potato_2"),
    (204, "find a SaltShaker_2"), (205, "find a SaltShaker_3"), (206, "find a Spoon_2"), (207, "find a Spoon_3"),
    # Extended variant (when present in alternate config):
    (208, "find a Fork_2"),  # Sometimes duplicate semantic; keep for safety if enumerated differently.
    (209, "find a Knife_2"), (210, "find a Lettuce_2"), (211, "find a PepperShaker_2"),
    (212, "find a SaltShaker_2"), (213, "find a SoapBottle_2"), (214, "find a SoapBottle_3"),
    (215, "find a Spatula_2"), (216, "find a Spatula_3"), (217, "find a WineBottle_2"),
]

ACTION_ID_TO_NAME: Dict[int, str] = {i: n for i, n in _ACTION_PAIRS}
# Reverse map (first occurrence wins; duplicates in extended tail are intentional).
ACTION_NAME_TO_ID: Dict[str, int] = {}
for _id, _name in _ACTION_PAIRS:
    ACTION_NAME_TO_ID.setdefault(_name, _id)

# Synonyms (normalized key -> canonical action name).
# Normalization rule in validator: lowercase + strip spaces + underscores.
SYNONYMS = {
    # Table variants
    "kitchentable": "find a DiningTable",  # If user hallucinated raw object reference
    "tablesurface": "find a DiningTable",
    "diningtable": "find a DiningTable",  # Already canonical, but safe
    # Generic attempts (map to Faucet / DiningTable when user adds descriptors)
    "faucetorwatersource": "find a Faucet",
    "watersource": "find a Faucet",
    # Sometimes model drops leading article
    "ladle": "find a Ladle",
    "pan": "find a Pan",
    # Mis-capitalization / spacing variations auto-resolve in normalization anyway.
}

__all__ = [
    "ACTION_ID_TO_NAME",
    "ACTION_NAME_TO_ID",
    "SYNONYMS",
]
