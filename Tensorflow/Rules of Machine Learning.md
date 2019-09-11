# Rules of Machine Learning
https://developers.google.com/machine-learning/guides/rules-of-ml/

## Terminology
- Metric: A number that you care about. May or may not be directly optimized.
- Pipeline: The infrastructure surrounding a machine learning algorithm. Includes gathering the data from the front end, putting it into training data files, training one or more models, and exporting the models to production.
- Click-through Rate: The percentage of visitors to a web page who click a link in an ad.

## Overview
- Make sure your pipeline is solid end to end.
- Start with a reasonable objective.
- Add common sense features in a simple way.
- Make sure that your pipeline stays solid.

## Rules of ML
- Rule #1: Don’t be afraid to launch a product without machine learning. If machine learning is not absolutely required for your product, don't use it until you have data.
- Rule #2: First, design and implement metrics. Before formalizing what your machine learning system will do, add a metric to track as much as possible in your current system.
- Rule #3: Choose machine learning over a complex heuristic. A complex heuristic is unmaintainable and you will find that the machine learned model is easier to update and maintain.
- Rule #4: Keep the first model simple and get the infrastructure right. Your simple model provides you with baseline metrics and a baseline behavior that you can use to test more complex models.
- Rule #5: Test the infrastructure independently from the machine learning. 
- Rule #6: Be careful about dropped data when copying pipelines. Often we create a pipeline by copying an existing pipeline (i.e., cargo cult programming ), and the old pipeline drops data that we need for the new pipeline.
- Rule #7: Turn heuristics into features, or handle them externally. Usually the problems that machine learning is trying to solve are not completely new. There are a bunch of rules and heuristics which give you a lift when tweaked with machine learning. There are four ways you can use an existing heuristic: 1.Preprocess using the heuristic. 2.Create a feature. 3.Mine the raw inputs of the heuristic. 4.Modify the label.
- Rule #8: Know the freshness requirements of your system. If you lose significant product quality if the model is not updated for a day, it makes sense to have an engineer watching it continuously.
- Rule #9: Detect problems before exporting models.
- Rule #10: Watch for silent failures. If you track statistics of the data, as well as manually inspect the data on occasion, you can reduce these kinds of failures.
- Rule #11: Give feature columns owners and documentation. If the system is large, and there are many feature columns, know who created or is maintaining each feature column. Although many feature columns have descriptive names, it's good to have a more detailed description of what the feature is, where it came from, and how it is expected to help.
- Rule #12: Don’t overthink which objective you choose to directly optimize. So, keep it simple and don’t think too hard about balancing different metrics when you can still easily increase all the metrics.
- Rule #13: Choose a simple, observable and attributable metric for your first objective. The ML objective should be something that is easy to measure and is a proxy for the "true" objective.
- Rule #14: Starting with an interpretable model makes debugging easier.
- Rule #15: Separate Spam Filtering and Quality Ranking in a Policy Layer.
- Rule #16: Plan to launch and iterate. So, as you build your model, think about how easy it is to add or remove or recombine features. Think about how easy it is to create a fresh copy of the pipeline and verify its correctness. Think about whether it is possible to have two or three copies running in parallel.
- Rule #17: Start with directly observed and reported features as opposed to learned features.
- Rule #18: Explore with features of content that generalize across contexts.
- Rule #19: Use very specific features when you can. With tons of data, it is simpler to learn millions of simple features than a few complex features. You can use regularization to eliminate the features that apply to too few examples.
- Rule #20: Combine and modify existing features to create new features in human-understandable ways. TensorFlow allow you to pre-process your data through transformations. The two most standard approaches are "discretizations" and "crosses".
- Rule #21: The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have.
- Rule #22: Clean up features you are no longer using.
- Rule #23: You are not a typical end user. Anything that looks reasonably near production should be tested further, either by paying laypeople to answer questions on a crowdsourcing platform, or through a live experiment on real users.
- Rule #24: Measure the delta between models. One of the easiest and sometimes most useful measurements you can make before any users have looked at your new model is to calculate just how different the new results are from production.
- Rule #25: When choosing models, utilitarian performance trumps predictive power. 
- Rule #26: Look for patterns in the measured errors, and create new features. The most important point is that this is an example that the machine learning system knows it got wrong and would like to fix if given the opportunity. If you give the model a feature that allows it to fix the error, the model will try to use it.
- Rule #27: Try to quantify observed undesirable behavior. If your issues are measurable, then you can start using them as features, objectives, or metrics. The general rule is "measure first, optimize second".
- Rule #28: Be aware that identical short-term behavior does not imply identical long-term behavior. Training-serving skew is a difference between performance during training and performance during serving. This skew can be caused by: 1.A discrepancy between how you handle data in the training and serving pipelines. 2.A change in the data between when you train and when you serve. 3.A feedback loop between your model and your algorithm.
- Rule #29: The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time. Even if you can’t do this for every example, do it for a small fraction, such that you can verify the consistency between serving and training.
- Rule #30: Importance-weight sampled data, don’t arbitrarily drop it! Importance weighting means that if you decide that you are going to sample example X with a 30% probability, then give it a weight of 10/3.
- Rule #31: Beware that if you join data from a table at training and serving time, the data in the table may change. The easiest way to avoid this sort of problem is to log features at serving time. If the table is changing only slowly, you can also snapshot the table hourly or daily to get reasonably close data. Note that this still doesn’t completely resolve the issue.
- Rule #32: Re-use code between your training pipeline and your serving pipeline whenever possible. Try not to use two different programming languages between training and serving. That decision will make it nearly impossible for you to share code.
- Rule #33: If you produce a model based on the data until January 5th, test the model on the data from January 6th and after. In general, measure performance of a model on the data gathered after the data you trained the model on, as this better reflects what your system will do in production.
- Rule #34: In binary classification for filtering (such as spam detection or determining interesting emails), make small short-term sacrifices in performance for very clean data.
- Rule #35: Beware of the inherent skew in ranking problems. When you switch your ranking algorithm radically enough that different results show up, you have effectively changed the data that your algorithm is going to see in the future. These approaches are all ways to favor data that your model has already seen: 1.Have higher regularization on features that cover more queries as opposed to those features that are on for only one query. This approach can help prevent very popular results from leaking into irrelevant queries. 2.Only allow features to have positive weights. Thus, any good feature will be better than a feature that is "unknown". 3.Don’t have document-only features. 
- Rule #36: Avoid feedback loops with positional features. The position of content dramatically affects how likely the user is to interact with it. Note that it is important to keep any positional features somewhat separate from the rest of the model because of this asymmetry between training and testing. Having the model be the sum of a function of the positional features and a function of the rest of the features is ideal.
- Rule #37: Measure Training/Serving Skew. There are several things that can cause skew in the most general sense: 1.The difference between the performance on the training data and the holdout data. 2.The difference between the performance on the holdout data and the "next day" data. 3.The difference between the performance on the "next-day" data and the live data.
- Rule #38: Don’t waste time on new features if unaligned objectives have become the issue. If the product goals are not covered by the existing algorithmic objective, you need to change either your objective or your product goals.
- Rule #39: Launch decisions are a proxy for long-term product goals. These metrics that are measureable in A/B tests in themselves are only a proxy for more long term goals: satisfying users, increasing users, satisfying partners, and profit, which even then you could consider proxies for having a useful, high quality product and a thriving company five years from now. The only easy launch decisions are when all metrics get better (or at least do not get worse).
- Rule #40: Keep ensembles simple. To keep things simple, each model should either be an ensemble only taking the input of other models, or a base model taking many features, but not both. 
- Rule #41: When performance plateaus, look for qualitatively new sources of information to add rather than refining existing signals.
- Rule #42: Don’t expect diversity, personalization, or relevance to be as correlated with popularity as you think they are.
- Rule #43: Your friends tend to be the same across different products. Your interests tend not to be.

