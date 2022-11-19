# LangVis
CS 279R Final Project

![demo](demo_img.png)

To install the dependencies, first run `pip install -e .` then run `pip install Flask` and `pip install SQLAlchemy`.

Write your prompt on the left panel, and hit generate button.

Hover your mouse on a generated word in the right panel to visualize its saliency map (redder color means more significant words in term of gradient-based saliency). 

The hovered word is highlighted in green.

Click on a word to see the other token that the word pays the most attention to (highlighted by a blue border).
