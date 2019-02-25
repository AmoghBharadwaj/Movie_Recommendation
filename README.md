#Movie Recommendation System

# Recommender Systems
At some point each one of us must have wondered where all the recommendations that Netflix, Amazon, Google give us, come from. We often rate products on the internet and all the preferences we express and data we share (explicitly or not), are used by recommender systems to generate, in fact, recommendations. The two main types of recommender systems are either collaborative or content-based filters: these two names are pretty self-explanatory, but let’s look at a couple of examples to better understand the differences between them. I will use movies as an example (because if I could, I would be watching movies/tv shows all the time), but keep in mind that this type of process can be applied for any kind of product you watch, listen to, buy, and so on


# Collaborative-based filter.
In order to work accurately, this type of filter needs rates, and not all users rate products constantly. Some of them barely or never rate anything! Another characteristic of this method is the diversity in the recommendations, that can be good or bad, depending on the case. For example, let’s say user A likes dystopian movies and dark comedy a lot. User B also enjoys dystopian movies but never watched dark comedy. The collaborative filter will recommend dark comedy shows to user B, based on the common taste that the two users have for dystopian movies. This scenario can go two ways: either user B finds out that he/she likes dark comedy a lot, and in that case, great, a lot of new things to watch on his/her list! Or, user B really enjoys a lighter comedy style, and in that case the recommendation has not been successful.

#Content-based filters
This type of filter does not involve other users if not ourselves. Based on what we like, the algorithm will simply pick items with similar content to recommend us.

In this case there will be less diversity in the recommendations, but this will work either the user rates things or not. If we compare this to the example above, maybe user B potentially likes dark comedy but he/she will never know, unless he/she decides to give it a try autonomously, because this filter will only keep recommending dystopian movies or similar. Of course there are many categories we can calculate the similarity on: in the case of movies we can decide to build our own recommender system based on genre only, or maybe we want to include director, main actors and so on.

These are not the only two types of existing filters, of course. There are also cluster or behavioral based for example, just to mention a couple more.