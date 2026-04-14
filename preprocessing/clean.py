# This file contains functions to clean the recipe data before feeding it into the model. 
# This includes normalizing ingredients and steps, as well as filtering out bad recipes.

def clean_ingredients(ingredients_list):
    """
    Normalize ingredients:
    - lowercase
    - strip whitespace
    - remove empty entries
    """
    cleaned = []

    for ing in ingredients_list:
        if not isinstance(ing, str):
            continue

        ing = ing.lower().strip()

        if ing:  # avoid empty strings
            cleaned.append(ing)

    return cleaned


def clean_steps(steps_list):
    """
    Clean recipe steps:
    - remove newline characters
    - strip whitespace
    - remove empty steps
    """
    cleaned = []

    for step in steps_list:
        if not isinstance(step, str):
            continue

        # remove newline characters
        step = step.replace('\n', ' ').strip()

        if step:
            cleaned.append(step)

    return cleaned


def is_valid_recipe(ingredients, steps):
    """
    Filter out bad recipes
    """
    if len(ingredients) < 3:
        return False

    if len(steps) < 2:
        return False

    return True