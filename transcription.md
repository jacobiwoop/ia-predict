# Transcription Vidéo

## Contenu avec horodatages

**[00:00]** In this video we will look at a trendline breakout trading strategy. We will first look at a simple

**[00:05]** but effective trend following strategy, then we will build a machine learning strategy to filter

**[00:10]** out false breakouts. This video will make use of the trendline functions I showed in this video.

**[00:15]** If you want to see the code for how these trendlines are drawn you can check that video out.

**[00:19]** The trendlines function binds the line that has the minimum distance to the input prices while

**[00:24]** also being above or below every price. In this video we're looking at trendline breakouts,

**[00:29]** But by design, the price will never be above or below these trend lines.

**[00:33]** To allow the price to break out, we do not include the current candle when we fit the trend lines.

**[00:38]** In other words, these trend lines are lagged by one candle. We extend these lines forward to the

**[00:43]** current candle. This way we can detect trend line breakouts. Let's start with the simple

**[00:48]** trend following strategy. We have this function which takes a closing price array and a look

**[00:53]** back parameter. We create output arrays for the support trend line, the resistance trend line,

**[00:59]** We loop through each price in the array. We get a recent window of prices, but we do not include the current price. This window is lagged by one candle. We use the function fitTrendlineSingle on the window of prices. This function was shown in the trendline video.

**[01:15]** It returns two sets of coefficients, a slope and intercept for both the support and resistance

**[01:21]** trendline. We use these slopes and intercepts to find the value of the trendlines for the

**[01:26]** current candle. We save the current values of the trendlines in these output arrays.

**[01:32]** Let's look at these two outputs on a plot along with the price. The light blue line is the hourly

**[01:37]** closing price of Bitcoin. The red and green lines are the current value of the support and resistance

**[01:42]** Inclure la ponctuation.

**[02:12]** If the price is above the current resistance value, then we set the signal to 1 for long.

**[02:17]** If it's below the current support value, then we set the signal to negative 1 for short,

**[02:22]** and otherwise we copy the signal from the previous candle.

**[02:26]** To test, we load in hourly Bitcoin data from a CSV.

**[02:29]** We call the trendline breakout function with the closing price and a lookback.

**[02:33]** I use 72, which is arbitrary.

**[02:36]** We save the returned arrays into our data frame.

**[02:38]** This is for plotting the bands we saw earlier.

**[02:41]** Then we get the next candle's log return. We can get the trading rules returns by multiplying this

**[02:47]** by the strategy signal. Then we compute the profit factor and plot the cumulative log return.

**[02:53]** It's decent, it has a profit factor of 1.035, but let's see how this performs across a wider

**[02:59]** range of parameters. Here are the profit factors across a wide range of lookbacks.

**[03:04]** There's a large spike in performance from 32 to 42, but my guess is that is just random luck.

**[03:10]** Beware of spikes in performance like this and the parameters of trading strategies.

**[03:15]** It is likely it won't carry forward to the future. But generally, the strategy has okay

**[03:19]** performance across most values. Not bad for having a position 100% of the time. I'm showing

**[03:24]** this simple strategy to show that the trendline breakout generally works, at least on Bitcoin.

**[03:30]** Now we'll move on to using a meta-labeling machine learning approach to filter false

**[03:34]** breakouts and hopefully get better results. But to do that, we need to define a more specific

**[03:38]** Here is a visualization of one of the breakout trades that we will consider.

**[03:44]** I'll continue using a 72 hour trendline and I'll focus on resistance or upper trendline breakouts

**[03:50]** for the rest of the video to keep things simple. The trade entry happens when the price closes

**[03:54]** above the trendline. I marked the candle that breaks out of the trendline in blue.

**[03:59]** For an exit I decided to use a 3 average true range or ATR stop loss and take profit centered

**[04:05]** I use a maximum hold period of 12 candles so if neither the stop loss or take profit is hit within 12 candles we exit the position The decision to use a 3 ATR stop loss and take profit and the 12 hour hold period is mostly arbitrary

**[04:22]** but we need definitive exit rules as we will use the outcome of these trades to build a label for our machine learning model.

**[04:28]** Our goal is to find features or indicators that are predictive of when this trade setup works and does not work,

**[04:34]** Then train a model with these indicators so we can make a prediction at the time of breakout

**[04:38]** to decide whether or not to take the trade. Let's look at the code for finding these trade setups

**[04:43]** and creating the dataset we will use later to train the machine learning model.

**[04:47]** This function finds the trade setups and records them in a dataset. It takes open high low close

**[04:53]** data, a look back for the trend line, a maximum holding period, stop loss, and take profit

**[04:58]** I set the ATR lookback to a week, or 168 since we're using hourly data here.

**[05:09]** We get the log closing price as a numpy array.

**[05:12]** Then we compute the average true range using log prices and convert it to a numpy array.

**[05:17]** Here I compute the normalized volume, that is volume divided by its median, and the ADX.

**[05:22]** We'll use these as features, but I'll talk about features later on.

**[05:25]** We create a pandas data frame to store the trade data, and a count to keep track of the trades added.

**[05:32]** We have these four variables for keeping track of the current trade.

**[05:35]** Price is for the stop loss and take profit, and HPI is the index of the maximum holding period for the current trade.

**[05:42]** We loop through each candle in the data, we get the recent window of prices, not including the current price, then fit the trend lines.

**[05:49]** We're again only considering the resistance breakout in this video, so I just project the upper trendline to the current bar.

**[05:56]** If we're not already in a trade and the close is above the resistance line, then we prepare an entry.

**[06:02]** We'd set the take profit and stop loss prices from the close plus and minus the average true range multiplied by their respective multipliers.

**[06:09]** We set HPI as the current index plus the holding period limit, and we set in trade to true.

**[06:15]** We had several pieces of data to the trade data frame on the current row, the index and price of the entry, the current average true range, take profit and stop loss prices, the hold period and slope intercept of the resistance trend line. Here onwards, we also add some features for the machine learning we'll do later. I'm going to skip over these for now. We'll come back to them later. Here we're still inside the loop through all the candles and this block manages the current trade. If we are, we check if it is time to exit.

**[06:43]** Comparing the close to the set take profit and stop loss, as well as checking the current loop index against the current holding period limit.

**[06:51]** If it is time to exit, we record the index and price of the exit, set intrade to false, and increment the trade count.

**[06:58]** After the loop is complete, we can compute the trade return as the exit price minus the entry price.

**[07:03]** We used logarithmic prices, so this return is roughly equivalent to a percentage.

**[07:08]** Later, when we're training a model, we'll need a label to train with.

**[07:11]** I chose a binary classification label. We label 1 if the trade had a positive return and 0 if it was negative.

**[07:18]** We return this label as dataY. We also pack the features in a separate data frame called dataX.

**[07:24]** The information in dataX and dataY are in the trade's data frame, but I just added them to separate data frames for convenience.

**[07:30]** Here's the performance of the trendline breakout with the 3ATR stop loss and take profit for hourly Bitcoin data from 2018 to the end of 2022.

**[07:39]** It is not good. The win rate is right at 50%, the profit factor is low at 1.02, the average trade is only about 0.05%.

**[07:49]** These results do not include slippage or fees, which would easily ruin these fairly positive results.

**[07:54]** But if we can build a model to predict when the strategy's trades work and do not work, we could potentially filter out some of the bad trades, turning this relatively bad strategy into a decent one.

**[08:05]** Let's consider some features to look at so a model can decide if one of these trendline breaks is worth trading.

**[08:10]** The common trait between each of these setups is the 72 hour resistance trendline breakout.

**[08:15]** We want features that can differentiate one setup from another.

**[08:18]** For a meta labeling model like this learning a specific type of trade I like to look at many of the trades and observe where they differ I know that sounds obvious but really you can learn a lot just by looking An obvious feature is the trendline slope Sometimes the trendline break

**[08:32]** happens in a downtrend, and sometimes it happens in an uptrend. It's a good idea to dedicate a

**[08:37]** portion of the data as an sample to assess features. Here I'll use the trades from 2018

**[08:42]** and 2019. Here's a scatter plot with the resistance slope on the x-axis and the trade return on the

**[08:48]** There isn't much of a relationship, but on average the trade returns are positive when the slope is positive and vice versa.

**[08:56]** There is a small positive Spearman correlation of 0.1.

**[08:59]** Perhaps this feature will be more useful in the presence of other features.

**[09:03]** It's common for traders to look at the volume when a breakout happens.

**[09:07]** Sometimes the volume is high when the trendline is broken, and sometimes it's low.

**[09:11]** Here's a scatterplot of the volume normalized by a rolling median and the trade returns.

**[09:15]** The relationship here isn't very strong, but it does have a small positive correlation.

**[09:20]** When the normalized volume is less than 1, that is the volume is less than the rolling median on the breakout, the average return is negative.

**[09:28]** We can look at how close the price is to the trendline. Sometimes the prices hug the trendline closely, and sometimes they are further away.

**[09:35]** This distance is minimized when the trendline is fit in the first place.

**[09:39]** Here is the average distance from a trendline versus the trade return.

**[09:43]** It has a small negative correlation. When the prices hug the trend lines more, trades tend to

**[09:47]** do better. Back to the code. This is the same trend line break dataset function we were looking

**[09:52]** at earlier. When we're getting an entry, we compute the slope indicator as the raw resistance

**[09:57]** slope divided by the average true range. The next indicator is the resistance error. We compute the

**[10:03]** value of the resistance line for each value in the window and subtract the prices from it. We

**[10:08]** Sum the error and average it by the look back window. Then divide this by the average true

**[10:13]** range to normalize it to volatility. This will measure how close the prices are on average to

**[10:19]** the resistance line. Similar to the resistance error, another feature I found effective was the

**[10:24]** maximum distance from the resistance line. This feature actually turns out to be the most

**[10:28]** informative. If there is a price in the window that we trained the trend line on that is very

**[10:33]** far away from it, then trades do not work very well. The next feature is the volume of the

**[10:38]** And the final feature is the ADX. I imagine most of you are familiar with this indicator.

**[10:45]** There's plenty of articles online about it if you're not. I typically include this indicator

**[10:49]** for meta-labeling models. It usually does pretty well. It measures the strength of the trend.

**[10:54]** I use the same lookback for the ADX as the trendline fit, so 72 in this case.

**[11:00]** These five features aren't necessarily the best ones or the only ones, but they are relevant to

**[11:05]** Inclure la ponctuation.

**[11:35]** Thank you for watching.

**[12:05]** But when we train many decision trees in a random forest, the noise has a tendency to cancel out.

**[12:11]** Let's move on to the walk forward code and see how a random forest works with the features we selected.

**[12:15]** We load in hourly Bitcoin data from a CSV, call the function trendline breakout dataset to get the trade setups.

**[12:22]** Then we call this function walk forward model.

**[12:24]** Walk forward model takes the closing price array.

**[12:27]** It also takes the trades data X and data Y that are all returned from the dataset function Inclure la ponctuation Inclure the output signals

**[12:49]** This one will be a binary signal. These will have the trades that the model selects.

**[12:54]** This other one will be the predicted probability of a profitable trade from the classifier.

**[12:59]** This variable nextTrain is the index that we will train the model at next.

**[13:04]** We initialize it as the train size.

**[13:06]** We have some variables to keep track of the current trade.

**[13:08]** This is all very similar to what we saw in the dataset function.

**[13:12]** We loop through each candle in the dataset.

**[13:14]** We first check if it is time to train.

**[13:16]** If it is, we find the indices of the trade's dataframe that happened within the last two years,

**[13:21]** or whatever you specified the train size to be.

**[13:24]** We check that the exit index is less than the current index.

**[13:27]** This way we don't cheat by leaking future information into the train set.

**[13:31]** We get the features and labels for the training indices.

**[13:35]** We create a random forest. I set max depth to 3 to control how deep the trees go.

**[13:40]** Ideally you should do a walk forward cross validation to set max depth,

**[13:44]** but I'm trying to keep this video from being too long,

**[13:46]** and in my experience a cross validation will almost always yield 2 or 3 as the best max depth.

**[13:50]** There's some randomness due to the bootstrap of the random forest.

**[13:53]** I set random state so the results will be the same across different runs.

**[13:57]** Inclure la ponctuation.

**[14:27]** Inclure la ponctuation.

**[14:57]** We save this probability to the probability signal, and add the predicted probability to

**[15:02]** the trades data frame. If the probability of the next trade is greater than 50%,

**[15:07]** we set signal to 1. We could use a higher threshold than 50%, which would result in

**[15:12]** less trades taken, but generally a higher performance. Then set the take profit,

**[15:17]** stop loss, and time limit for the current trade. Finally, we increment the trade eye index,

**[15:22]** and that's it for the walk forward code. Here's the performance of the trades that

**[15:26]** The model picked and all the trades with no filtering. The first two years were used for

**[15:31]** training, so it's flat here. The model appears to have done its job. It outperformed the underlying

**[15:36]** strategy. The trades from the underlying strategy were in the market 30% of the time, while the

**[15:40]** model's chosen trades were in the market 20% of the time. The model boosted the win rate slightly

**[15:45]** and more than doubled the average trade. The improved strategy did struggle in late 2021

**[15:50]** Inclure la ponctuation.

**[16:20]** Thank you for watching.

