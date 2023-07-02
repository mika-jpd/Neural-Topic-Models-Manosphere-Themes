# Application of Neural Topic Models for Exploration of Themes in Online Communities

Analysis of manosphere forums using Deep Neural Network based Topic Models

*I will be reorganizing the folder structure, refactoring the code and adding comments so that the project is easier to understand once I have the time !*

## Project Organization
**Note 1: as our project deals with online masculine communities, some of the posts mentioned in this project contain disturbing material.**
**Note 2: Data and model weights folders are missing !**

This is the final project for the course Machine Learning Practical (INFR11132) taken during my masters at the University of Edinburgh which I completed with Dmitrii Tiron and Nicholas Zhang.

## Abstract 

In recent years, online male support groups, which are part of the manosphere, have been put in the spotlight because of their rise in misogynistic language, calls to violence, and growing popularity. However, the unstructured nature of online communities means that it is challenging to interpret. Due to recent advances in deep learning architectures for text, Neural Topic Models provide a potential solution to large-scale thematic analysis of textual data over classical methods like LDA. An obstacle in the way of large-scale adoption of Topic Modelling in Social Sciences is finding ways to evaluate a model's outputs, for which human review remains a top choice. The development of rigorous methodology for human evaluation, however, has received little attention. In our project we explore the use of neural topic models and compare their performance to LDA using topic coherence and topic diversity as quantitative metrics. Furthermore, we then treat the themes found by social scientists as gold standard in our evaluation, and present two methods to evaluate results of our topic model: topic and gender keyword association tests and document topic comparison. 

We find that we can validate that some neural topics models are able to perform better than LDA on social media data and achieve good scores on standard metrics. Furthermore, using sentence embeddings not only helps models perform well, but also o↵ers flexibility and a wider range of analysis methods. In the Keyword-Gender Association Test from Section we were unable to consistently replicate the gendered association found by Vallerga & Zurbriggen, 2022 in their analysis. We identified that future work could explore the syntactic role gendered terms in representative documents to identify gender roles instead of relying only on word co-occurence. Finally, in the Document-Topic Comparison, BERTopic was unable to find the same topics for representative texts of themes in (Vallerga & Zurbriggen, 2022).
