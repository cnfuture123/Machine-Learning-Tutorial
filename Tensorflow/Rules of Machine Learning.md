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

