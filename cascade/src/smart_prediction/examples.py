"""Example documents used by the smart-prediction prototype."""

# ---------------------------------------------------------------------------
# Poems
# ---------------------------------------------------------------------------


POEM_THREE_STANZAS = (
    "Roses are red,\n"
    "Violets are blue,\n"
    "Sugar is sweet,\n"
    "And so are you.\n\n"
    "Sunflowers are yellow,\n"
    "Daisies are white,\n"
    "Nature paints colors,\n"
    "Morning to night.\n\n"
    "Lilies smell fragrant,\n"
    "Tulips so bright,\n"
    "Flowers give comfort,\n"
    "And hearts delight."  # no trailing newline on purpose
)

# Remove the *middle* stanza so that stanza 1 is immediately followed by 3.
POEM_MISSING_MIDDLE_STANZA = (
    "Roses are red,\n"
    "Violets are blue,\n"
    "Sugar is sweet,\n"
    "And so are you.\n\n"
    # (middle stanza skipped)\n"
    "Lilies smell fragrant,\n"
    "Tulips so bright,\n"
    "Flowers give comfort,\n"
    "And hearts delight."
)
