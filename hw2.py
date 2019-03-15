import pandas as pd
import numpy as np
import wbdata
import datetime
from scipy import stats
wbdata.get_source()
wbdata.get_indicator(source=12)

#date setting
data_date = (datetime.datetime(2010, 1, 1), datetime.datetime(2011, 1, 1))

#Search Indicators
for i in range(1, np.size(x)):
    print(x[i])
    x = wbdata.search_indicators("per capita")
    for i in range(1, np.size(x)):
        print(x[i]['sourceNote'])
del(x)
#Countries and indicatiors assigment
countries = [i['id'] for i in wbdata.get_country(incomelevel="HIC", display=False)]
indicators1 = { "NY.GDP.PCAP.PP.KD": "gdppc","IC.REG.COST.PC.MA.ZS":"LAC Equity Lab","IC.REG.COST.PC.MA.ZS": "Cost of business start-up procedures, male (% of GNI per capita)"}
indicators2 = { "NY.GDP.PCAP.PP.KD": "gdppc","SH.XPD.CHEX.PP.CD" :"World Development Indicators"}
df = wbdata.get_dataframe(indicators1, country=countries, convert_date=True)
df2=  wbdata.get_dataframe(indicators2, country=countries, convert_date=True)
#delete na
df.dropna()
df2.dropna()

df2.describe()
df.describe()

#write csv and read
df.to_csv('df.csv')
df2.to_csv('df2.csv')


df=pd.read_csv('df.csv')
df2=pd.read_csv('df2.csv')

df["uniqid"]=(df.country +df.date)

df2["uniqid"]=(df2.country +df2.date)

dfmerge = df.merge(df2,on="uniqid")

dfmerge.to_csv("dfmerge")
dfmerge=pd.read_csv("dfmerge.csv")

dfmerge={"country":dfmerge["country_x"],"gdppc":dfmerge["gdppc_x"],"World Development Indicators":dfmerge["World Development Indicators"],"CoBstart_up or. %GNI":dfmerge["Cost of business start-up procedures, male (% of GNI per capita)"]}
dfmerge=pd.DataFrame(data=dfmerge)
dfmerge=dfmerge.dropna()

x=dfmerge[['World Development Indicators','CoBstart_up or. %GNI']]
y=dfmerge['gdppc']
def A (x,y):
    import numpy as np
    import pandas as pd
    # NaN remove from df.
    x.dropna()
    y.dropna()
    # Covert to matrix
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    y=y.transpose()
    # Calc. Coefficient Matrix
    beta_hat = (np.linalg.inv(x.transpose() * x) * x.transpose()) * y
    y_hat = x * beta_hat
    # Calc. Error Matrix
    error = y - y_hat
    sigma_sq = (error.transpose() * error) / (y.shape[0] - x.shape[1] - 1)
    sigma = np.sqrt((error.transpose() * error) / (y.shape[0] - x.shape[1] - 1))
    var_b_hat = (sigma_sq.item(0)) * np.linalg.inv(x.transpose() * x)

    # Create T and P Values Matrix
    t_values= np.zeros((x.shape[1],1))
    p_values=np.zeros((x.shape[1],1))

    for i in range(0,x.shape[1]-1):
        t_values[i,0]=beta_hat[i,0]/sigma
    for i in range(0, x.shape[1]-1):
        p_values[i,0] = stats.t.sf(np.abs(t_values.item(i)), x.shape[1] - 1) * 2
    # Print fuctions
    for i in range(0, beta_hat.size):
        print( str(i) + ". Coefficient is " + str(beta_hat.item(i).__round__(4)) +" || T value is " +str(t_values.item(i).__round__(4)) +" P Values is " +str(p_values.item(i).__round__(4)))
    mean_x, mean_y = np.mean(x), np.mean(y)
    SSR = np.sum(np.square(y_hat - mean_y))
    print("SSR=" + str(SSR.round(2)))
    SSRes = y.transpose() * y - beta_hat.transpose() * x.transpose() * y
    print("SSRes = " + str(SSRes.item(0).__round__(2)))
    F = (SSR / x.shape[1]) / (SSRes / (y.shape[0] - x.shape[1] - 1))
    print("F = " + str(F.item(0).__round__(2)))
    SST = y.transpose() * y - np.square(mean_y) / y.shape[0]
    print("SST = " + str(SST.item(0).__round__(2)))
    R_sq = 1 - (SSR / (y.shape[0] - x.shape[1] - 1)) / (SST / (y.shape[0] - 1))
    print("R_sq = " + str(R_sq.item(0).__round__(2)))
