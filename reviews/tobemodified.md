

##### 2.4

Explain a bit more detail. (See enote)

##### 2.5

Add an "expected gender direction" to the figure, and declare "the direction found by Densray is closer to the expected direction than Hard Debiasing."

##### 3.1

one of the gendered words -> at least one of the gendered words
I had checked the code, we use 'at least one', it's a writing mistake here.

##### 3.2

OCCTMP (occupation templates)

##### 3.2.3

Add a comment "Since the deviation already exists in the data of these tasks, the overall performance will decrease after debiasing. Our expectation here is to cause less damage to the model performance after debiasing."

##### 4.1

Modify the writing for better understanding (See enote)

##### 4.4

Cite Bau et al (2018). (See enote)

##### 4.5

Explain why we debias on English and Chinese, but not French which marks gender. (See enote)

##### 5.2

cite Prost et al. (2019) (See enote)

##### Related Works

Add 5 new papers to the bib (the top 5).
Bau et al (2019) is used in 4.4. Prost et al. (2019) is used in 5.2. The other three needs to be cited:

###### Gonen and Goldberg (2019):  

1. This paper showed that only remove bias from the gender direction is not good enough, bias also come from the association to other implicitly gendered terms. I think the scope of this paper should just cover gender direction, Gonen and Goldberg (2019) can be used in 5.2 to show the limitations of our work. 
2. They also proposed some experiments (biased words clustering, neighbours, gender classifier) to show the remaining bias after debiasing. I think the first two are not so good to be applied on contextual embedding, the third shared the same idea with OCCTMP.
3. In another aspect, intuitively on each BERT layer, the context of a token will contribute some 'association' to the gender direction of that token, so I think this is match with our approach that applied densray to all bert layers (can answer the question from reviwer3). 

###### Vanmassenhove et al (2018), Moryosseft et al (2019): 

This two paper showed that adding gender information to NMT system can help boost the performance (the second paper is based on the first one). I think they are not so related to our work. Maybe we can mentioned them in section 1, to show the importance of  gender bias in NLP applications?

##### References

already change some arxiv papers to their published version. 

##### Others

1. No enough space. Need to remove something or move to the appendix (I suggest 4.1 subsection 'Number of Training Samples'). 
2. Publish the code (and dataset) on github.

##### Interesting points from the reviewers but Not in the scope (Don't need to do):

1. the difference between bert base and large
2. how to reduce gender bias in French? what's the format of gender bias in French?
3. Debias for racial/religious
4. consider other training data pairs like 'waiter' vs 'waitress' or 'host' vs 'hostess'



