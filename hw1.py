import time
class Portfolio(object):

    cash = 0
    stocks = []
    mutualFunds = []
    transactions = []
    def addCash(self, cash):
        self.cash = self.cash + cash
        self.transactions.append("Deposit: +" + str(cash) + " | " + "Balance: " + str(self.cash) +" | " + self.getTime())

    def buyStock(self, share, stock):
        ownedStock = OwnedStock(share, stock)
        self.stocks.append(ownedStock)
        self.cash = self.cash - stock.price * share
        self.transactions.append(str(share) + " unit " + stock.symbol + " Stock bought " + "Balance: " + str(self.cash)+" | " + self.getTime())

    def buyMutualFund(self, share, mutualFund):
        ownedMutualFund = OwnedMutualFund(share, mutualFund)
        self.mutualFunds.append(ownedMutualFund)
        self.transactions.append(str(share) + " unit " + mutualFund.symbol + " Mutual Fund bought " + "Balance: " + str(self.cash)+" | " + self.getTime())

    def withdrawCash(self, cash):
        self.cash = self.cash - cash
        self.transactions.append("withdraw: -" + str(cash) + " | " + "Balance: " + str(self.cash)+" | " + self.getTime())

    def sellMutualFund(self, symbol, share):
        for mutualFundObject in self.mutualFunds:
            if mutualFundObject.mutualFund.symbol == symbol:
                mutualFundObject.share = mutualFundObject.share - share
                self.transactions.append(str(mutualFundObject.share) + " unit " + mutualFundObject.mutualFund.symbol + " Mutual Fund sold " + "Balance: " + str(self.cash)+" | " + self.getTime())
                break

    def sellStock(self, symbol, share):
        for stockObject in self.stocks:
            if stockObject.stock.symbol == symbol:
                stockObject.share = stockObject.share - share
                self.cash = self.cash + stockObject.stock.price * share
                self.transactions.append(str(share) + " unit " + stockObject.stock.symbol + " Stock sold " + "Balance: " + str(self.cash)+" | " + self.getTime())
                break

    def history(self):
        print("\n".join(self.transactions))

    def __str__(self):
        accountString = "Accounts: "+ '\n'
        cashString = "cash: $" + str(self.cash) + "\n"
        stockString = "stocks: "
        for stockObject in self.stocks:
            stockString = stockString + str(stockObject.share) + " " + stockObject.stock.symbol + "\n"

        mutualFundString = "mutual funds: "
        for mutualFundObject in self.mutualFunds:
            mutualFundString = mutualFundString + str(mutualFundObject.share) + " " + mutualFundObject.mutualFund.symbol + "\n"
        return accountString + cashString + stockString + mutualFundString

    def getTime(self):
        return time.ctime(time.time())
class OwnedStock(object):
    share = 0
    stock = None;

    def __init__(self, share, stock):
        self.share = share
        self.stock = stock

class OwnedMutualFund(object):
    share = 0
    mutualFund = None;

    def __init__(self, share, mutualFund):
        self.share = share
        self.mutualFund= mutualFund

class Stock(object):
    price = 0
    symbol = 0

    def __init__(self, price, symbol):
        self.price = price
        self.symbol = symbol


class MutualFund(object):
    symbol = 0

    def __init__(self, symbol):
        self.symbol = symbol

client1 = Portfolio()
client1.addCash(300)

asusStock = Stock(30, "ASUS")
microsoftStock = Stock(10, "MIC")

client1.buyStock(3, asusStock)
client1.buyStock(1, microsoftStock)

mf1 = MutualFund("BRT")
mf2 = MutualFund("GHT")

client1.buyMutualFund(7.3, mf1)
client1.buyMutualFund(3, mf2)

client1.sellMutualFund("GHT", 2)
client1.sellStock("ASUS", 2)
client1.withdrawCash(260)

print(client1)

client1.history()