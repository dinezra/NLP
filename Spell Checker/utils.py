def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace("\n", " ")

    text = text.replace("'s", "")
    text = text.strip()
    return text  
  