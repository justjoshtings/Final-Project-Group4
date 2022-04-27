# Woby Tales ![woby](https://emojipedia-us.s3.amazonaws.com/source/skype/289/ghost_1f47b.png)
Human-machine interactive storytelling engine finetuned for scary/creepy/chilling stories. Finetuned on GPT2, GPT-NEO, and a custom pretrained GPT2 transformer networks.

#### Woby can generate and continue its own scary story.
![woby_only](https://github.com/justjoshtings/Final-Project-Group4/blob/main/Code/results/woby_only.gif)

#### Woby also allows users to begin the story and continue along.
![woby_human](https://github.com/justjoshtings/Final-Project-Group4/blob/main/Code/results/woby_human.gif)

# How to Run
See [App Execution](https://github.com/justjoshtings/Final-Project-Group4/blob/main/Code/README.md#app-execution) for details

# Data Sources
Stories were sourced from the following sub-reddits

Number | Subreddit | Link 
| --- | --- | --- |
| 1. | r/nosleep | [Link](https://www.reddit.com/r/nosleep/) | 
| 2. | r/stayawake | [Link](https://www.reddit.com/r/stayawake/) | 
| 3. | r/DarkTales | [Link](https://www.reddit.com/r/DarkTales/) | 
| 4. | r/LetsNotMeet | [Link](https://www.reddit.com/r/LetsNotMeet/) | 
| 5. | r/shortscarystories | [Link](https://www.reddit.com/r/shortscarystories/) | 
| 6. | r/Thetruthishere | [Link](https://www.reddit.com/r/Thetruthishere/) | 
| 7. | r/creepyencounters | [Link](https://www.reddit.com/r/creepyencounters/) | 
| 8. | r/truescarystories | [Link](https://www.reddit.com/r/TrueScaryStories/) | 
| 9. | r/Glitch_in_the_Matrix | [Link](https://www.reddit.com/r/Glitch_in_the_Matrix/) | 
| 10. | r/Paranormal | [Link](https://www.reddit.com/r/Paranormal/) | 
| 11. | r/Ghoststories | [Link](https://www.reddit.com/r/Ghoststories/) | 
| 12. | r/libraryofshadows | [Link](https://www.reddit.com/r/libraryofshadows/) | 
| 13. | r/UnresolvedMysteries | [Link](https://www.reddit.com/r/UnresolvedMysteries/) | 
| 14. | r/TheChills | [Link](https://www.reddit.com/r/TheChills/) | 
| 15. | r/scaredshitless | [Link](https://www.reddit.com/r/scaredshitless/) | 
| 16. | r/scaryshortstories | [Link](https://www.reddit.com/r/scaryshortstories/) | 
| 17. | r/Humanoidencounters | [Link](https://www.reddit.com/r/Humanoidencounters/) | 
| 18. | r/DispatchingStories | [Link](https://www.reddit.com/r/DispatchingStories/) | 

See [Data Acquisition section for more details.](https://github.com/justjoshtings/Final-Project-Group4/blob/main/Code/README.md#data-acquisition)

# Contents
1. **Code**: This directory holds all relevant code for data acquisition, preprocessing, model building/training, evaluation, and front end.
2. **Group-Proposal**: Group proposal description for project.
3. **.gitignore**: gitignore file
4. **LICENSE**: license description
5. **README.md**: readme file
6. **requirements.txt**: python requirements

# References

## Similar Projects/Products
1. [This AI Writes Horror Stories, And They’re Surprisingly Scary, 2017](https://www.fastcompany.com/90148966/this-ai-writes-horror-stories-and-theyre-surprisingly-scary)
2. [AI Dungeon](https://play.aidungeon.io/main/home)

## Background
3. [Conditional Text Generation for Harmonious Human-Machine Interaction, Guo et al., 2020](https://arxiv.org/pdf/1909.03409.pdf)
4. [Transformers: State-of-the-Art Natural Language Processing, Wolf et al., 2020](https://aclanthology.org/2020.emnlp-demos.6/)
```
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
5. [Transformer-based Conditional Variational Autoencoder for Controllable Story Generation, Fang et al., 2021](https://arxiv.org/pdf/2101.00828.pdf)
6. [https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1214&context=scschcomdis, Araz, 2020](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1214&context=scschcomdis)
7. [ Improving Neural Story Generation by Targeted Common Sense Grounding, hhmao et al., 2019](https://aclanthology.org/D19-1615.pdf)
8. [Evaluation of Text Generation: A Survey, Celikyilmaz et al., 2021](https://arxiv.org/pdf/2006.14799.pdf)
