"""Goal DSL package — atoms, combinators, types, and species/region constants.

Usage:

    from pokemon_planner import goals as g

    g.catch(g.ODDISH)
    g.then(g.beat(g.BROCK), g.catch(g.BELLSPROUT))
    g.forall((g.BULBASAUR, g.CHARMANDER, g.SQUIRTLE), lambda p: g.catch(p))
"""
from pokemon_planner.goals.atoms import (
    and_,
    beat,
    catch,
    evolve,
    forall,
    have_item,
    level,
    or_,
    reach,
    then,
)
from pokemon_planner.goals.dsl import (
    And,
    Atom,
    Forall,
    Goal,
    Or,
    Then,
)


# ---- Convenience constants ----
# Names mirror their KB entries. Importing them as constants lets users write
# g.catch(g.ODDISH) instead of g.catch("ODDISH"). The values are just strings
# that the predicate evaluator looks up via the species_id_lookup table.

ODDISH = "ODDISH"
PIDGEY = "PIDGEY"
RATTATA = "RATTATA"
PIKACHU = "PIKACHU"
BULBASAUR = "BULBASAUR"
CHARMANDER = "CHARMANDER"
SQUIRTLE = "SQUIRTLE"
CHARMELEON = "CHARMELEON"
CHARIZARD = "CHARIZARD"

BROCK = "BROCK"
MISTY = "MISTY"

ROUTE_1 = "ROUTE_1"
ROUTE_5 = "ROUTE_5"
PEWTER_CITY = "PEWTER_CITY"
CERULEAN_CITY = "CERULEAN_CITY"

POKEBALL = "POKEBALL"
POTION = "POTION"
BICYCLE = "BICYCLE"


__all__ = [
    # Combinators
    "and_", "beat", "catch", "evolve", "forall", "have_item",
    "level", "or_", "reach", "then",
    # Types
    "And", "Atom", "Forall", "Goal", "Or", "Then",
    # Constants
    "ODDISH", "PIDGEY", "RATTATA", "PIKACHU",
    "BULBASAUR", "CHARMANDER", "SQUIRTLE",
    "CHARMELEON", "CHARIZARD",
    "BROCK", "MISTY",
    "ROUTE_1", "ROUTE_5", "PEWTER_CITY", "CERULEAN_CITY",
    "POKEBALL", "POTION", "BICYCLE",
]
