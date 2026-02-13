from cobra import Model, Reaction
from typing import cast

Numerical = int | float



def clone_with_modified_bounds(model: Model, bounds_dict: dict[str, tuple[int, int]]) -> Model:
    '''Returns a copy of the given COBRA model with reaction bounds updated. The reactions specified in "bounds_dict" are replaced by the provided (lower, upper) tuples.
    The original model is not modified.'''
    new_model = model.copy()

    for reaction_id, bounds in bounds_dict.items():
        reaction = cast(Reaction, new_model.reactions.get_by_id(reaction_id))
        reaction.bounds = bounds

    return new_model


def add_reaction_from_string(model: Model, reaction_name: str, reaction_equation: str, 
                             lower_bound: Numerical | None = None, upper_bound: Numerical | None = None) -> None:
    '''Adds a reaction to a COBRA model using a reaction equation string.

    The reaction is created, added to the model, and then populated from
    a stoichiometric equation (e.g. 'A_e -> A_c'). Optional lower and upper
    bounds can be specified.'''

    reaction = Reaction(reaction_name)
    model.add_reactions([reaction])
    reaction.build_reaction_from_string(reaction_equation)

    if lower_bound is not None:
        reaction.lower_bound = lower_bound

    if upper_bound is not None:
        reaction.upper_bound = upper_bound


def add_reactions_from_dictionary(model: Model, reaction_dict: dict[str, dict]) -> None:
    '''Adds multiple reactions to a COBRA model from a reaction specification dictionary.
    Each entry must define an 'equation' and may optionally define bounds.'''

    for name, reaction_attributes in reaction_dict.items():
        if 'equation' not in reaction_attributes:
            raise ValueError(f"Reaction '{name}' missing 'equation'")

        add_reaction_from_string(
            model,
            reaction_name=name,
            reaction_equation=reaction_attributes['equation'],
            lower_bound=reaction_attributes.get('lowerbound'),
            upper_bound=reaction_attributes.get('upperbound'),
        )
