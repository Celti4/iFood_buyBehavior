#Data Viz Graphs for iFood Buy Behavior


# GRAPHS
# DESCRIPTIVE ANALYSYS

plt.figure(figsize = (24,8))
sns.set(rc={"axes.facecolor":"#283747", "axes.grid":False,'xtick.labelsize':10})
sns.lineplot(x = "Year-Month", y = "Average Spend", data = dfMesNormal, linewidth = 2, label = "Before Covid", markers = True)
sns.lineplot(x = "Year-Month", y = "Average Spend", data = dfMesCovid, linewidth = 2, label = "After Covid", markers = True)
sns.lineplot(x = "Year-Month", y = "Mean", data = dfMesNormal, linewidth = 1, label = "Average Before Covid", color = "yellow", linestyle = "--")
sns.lineplot(x = "Year-Month", y = "Mean", data = dfMesCovid, linewidth = 1, label = "Average After Covid", linestyle = "--")
plt.title("Average Ticket per Month")
plt.ylabel("R$")
plt.xlabel("Date")
plt.legend(facecolor= 'white' , fontsize='large' , edgecolor = 'black' ,shadow=True)

plt.figure(figsize = (25,8))
sns.lineplot(x = "Year-Month", y = "Orders", data = dfMesNormal, linewidth = 2, label = "Before Covid")
sns.lineplot(x = "Year-Month", y = "Orders", data = dfMesCovid, linewidth = 2, label = "After Covid")
sns.lineplot(x = "Year-Month", y = "AverageOrders", data = dfMesNormal, linewidth = 1, label = "Average Before Covid", color = "yellow", linestyle = "--")
sns.lineplot(x = "Year-Month", y = "AverageOrders", data = dfMesCovid, linewidth = 1, label = "Average After Covid", linestyle = "--")
plt.title("Monthly Number of orders")
plt.ylabel("Number of Orders")
plt.xlabel("Date")
plt.legend(facecolor= 'white' , fontsize='large' , edgecolor = 'black' ,shadow=True)



plt.figure(figsize = (25,8))
sns.lineplot(x = "Year-Month", y = "Total", data = dfMesNormal, linewidth = 2, label = "Before Covid")
sns.lineplot(x = "Year-Month", y = "Total", data = dfMesCovid, linewidth = 2, label = "After Covid")
sns.lineplot(x = "Year-Month", y = dfMesNormal.Total.mean(), data = dfMesNormal, linewidth = 1, label = "Average Before Covid", color = "yellow", linestyle = "--")
sns.lineplot(x = "Year-Month", y = dfMesCovid.Total.mean(), data = dfMesCovid, linewidth = 1, label = "Average After Covid", linestyle = "--")
plt.title("Total Spent per Month")
plt.ylabel("R$")
plt.xlabel("Date")
plt.legend(facecolor= 'white' , fontsize='large' , edgecolor = 'black' ,shadow=True)



plt.figure(figsize = (12,12))
sns.displot(x = "valor", data = df, hue = "Pandemia", kind = "kde")
plt.title("R$ Spent Probability distribution")
plt.xlabel("R$")
plt.legend(facecolor= 'white' , fontsize='large' , edgecolor = 'black' ,shadow=True)


sns.displot(x = "difTime", data = dfCovid, kind = "kde")





# NPD

plt.figure(figsize = (16,9))
sns.heatmap(Markov_chain_prob, cmap = "coolwarm", annot = True)
plt.yticks(rotation = 0)

plt.figure(figsize = (12,12))
sns.displot(x = resultsList, bins = 10)
plt.title("Correct predictions Distribution over 1000 simulations")
plt.xlabel("Correct Prediction Markov Model")
plt.ylabel("Frequency")


plt.figure(figsize = (12,8))
plt.step(x = "Round", y = "modelScore", data = dfGraphReference, label = "Model Prediction")
plt.step(x = "Round", y = "perfectScore", data = dfGraphReference, label = "Perfect Prediction")
plt.step(x = "Round", y = "randomScore", data = dfGraphReference, label = "Random day prediction")
plt.title("Accuracy from each model: Random, Perfect and Markov Chain")
plt.ylabel("Number of Correct Predictions")
plt.xlabel("Buy index")
plt.legend(facecolor= 'white' , fontsize='large' , edgecolor = 'black' ,shadow=True)
