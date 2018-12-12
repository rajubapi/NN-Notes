# A guide to Contribution

## Prerequisites

* You must be familiar with git. A basic knowledge of the commands: `push`, `pull`, `add`, `commit` would suffice

* Familiarity with Jupyter Notebook

* Basic knowledge of python. Mind you, the extent of programming knowlegde that you might require depends on the topic that you choose. In general, you will have to use `numpy` and `matplotlib`

* You might have to write mathematical equations, these are written in Latex. Therefore, you are expected to be familiar with the syntax.

* Knowledge of Markdown style writing and basic HTML might help you to represent your content better

## Rubric

* Create a new branch for each sub-topic. They will later be merged into their own respective branches: Perceptron, MLFFNN, and SVM

* Each branch would be prefixed with your name to avoid confusions eg: `sai/hot-stuff`, here hot stuff is something that you are working on. You can let it be the the name of the topic that you are working on

* Each branch will house only a single directory. All the files must reside inside it

* All your code for drwaing a figure would reside in a separate `.py` file.

* These `figure` files created then shall be imported into the jupyter notebook. You may choose to write a single function called `display()`  with necessary parameters to display the figure. This way the notebook will house only the important stuff. Peeps can have a peek at the code if they choose so...

## Commit message format

* The commits must be in imperative mood. The best way to obtain it is like so:

Consider that you have made a commit which does the follows:

```bash
This commit will add the draw function
```

Your commit message will look like so:

```bash
Add the draw function
```

Just chuck the `This commit will` part.

* Captalise the first letter of the commit

* Do not end the subject line with a peroid

## Final note

You can pick up any topic from the TODO list. You might conisder dropping a message on the Whatsapp group, so that other know that you are working on it. 

**Do not forget to sign your commits. This is the only way to ensure the authenticity of the author, that, is you.** [Refer this](https://pedrorijo.com/blog/git-gpg/)