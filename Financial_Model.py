#================================================================
# Final Project: Financial Modeling package
#================================================================

#%% All financial modeling functions

# -----------------------------------------------------------
# Import packages 

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import numpy_financial as npf 
from sklearn.linear_model import LinearRegression



# -----------------------------------------------------------
# [0] Useful functions 

def tab_clean(tab):
    tab.columns = tab.loc[2]
    tab.drop(index = [0,1,2,3], inplace = True)
    tab.drop(columns = np.nan, inplace = True)
    tab.dropna(how = 'all', inplace = True)
    tab.replace('—', 0 , inplace = True)
    tab.set_index('In Millions of USD except Per Share', inplace = True)
    return(tab)


def rewrite_bs(bs_tab):
    fs = pd.DataFrame(columns = bs_tab.columns)

#% Rewrite Balance sheet
    fs.loc['Balance Sheet'] = np.nan
    fs.loc['Cash and marketable securities'] = bs_tab.loc['  + Cash, Cash Equivalents & STI']
    fs.loc['Current assets'] = (bs_tab.loc['  + Accounts & Notes Receiv']
                               + bs_tab.loc['  + Inventories']
                               + bs_tab.loc['  + Other ST Assets'])
    fs.loc['Fixed assets at cost'] = bs_tab.loc['    + Property, Plant & Equip']
    fs.loc['Accumulated depreciation'] = - bs_tab.loc['    - Accumulated Depreciation']
    fs.loc['Net fixed assets'] = bs_tab.loc['  + Property, Plant & Equip, Net']
    fs.loc['Long-term investments'] = bs_tab.loc['  + LT Investments & Receivables']
    fs.loc['Other long-term assets'] = bs_tab.loc['  + Other LT Assets']
    fs.loc['Total assets'] = bs_tab.loc['Total Assets'].iloc[1]
    fs.loc['Current liabilities'] = (bs_tab.loc['  + Payables & Accruals']
                                    + bs_tab.loc['  + Other ST Liabilities'])
    fs.loc['Debt'] = bs_tab.loc['  + ST Debt'] + bs_tab.loc['  + LT Debt']
    fs.loc['Other long-term liabilities'] = bs_tab.loc['  + Other LT Liabilities']
    fs.loc['Total liabilities'] = bs_tab.loc['Total Liabilities']

    fs.loc['Preferred equity'] = bs_tab.loc['  + Preferred Equity and Hybrid Capital']
    fs.loc['Minority interest'] = bs_tab.loc['  + Minority/Non Controlling Interest']
    fs.loc['Common stock'] = bs_tab.loc['  + Share Capital & APIC'] 
    fs.loc['Treasury Stock'] = - bs_tab.loc['  - Treasury Stock'] 
    fs.loc['Accumulated retained earnings'] = bs_tab.loc['  + Retained Earnings']
    fs.loc['Other equity'] = bs_tab.loc['  + Other Equity']  
    fs.loc['Equity'] = (bs_tab.loc['Equity Before Minority Interest'] - 
                        bs_tab.loc['  + Preferred Equity and Hybrid Capital'])
    fs.loc['Total liabilities and equity']  = bs_tab.loc['Total Liabilities & Equity']
    
    return(fs)


def rewrite_is(is_tab):
    fs = pd.DataFrame(columns = is_tab.columns)
    fs.loc['Income Statement'] = np.nan
    fs.loc['Sales'] = is_tab.loc['Revenue']
    fs.loc['Cost of goods and sold'] = - is_tab.loc['  - Cost of Revenue']
    fs.loc['Depreciation'] = - is_tab.loc['Depreciation Expense']
    fs.loc['Other operating costs'] = -(is_tab.loc['Gross Profit'] 
                                        - is_tab.loc['Depreciation Expense'] 
                                        - is_tab.loc['Operating Income (Loss)'])
    fs.loc['Operating income'] = is_tab.loc['Operating Income (Loss)']
    fs.loc['Interest payments on debt'] = - is_tab.loc['    + Interest Expense']
    fs.loc['Interest earned on cash and marketable securities'] = is_tab.loc['    - Interest Income']
    fs.loc['Other non-operating costs'] = - (is_tab.loc['Operating Income (Loss)']
                                             - is_tab.loc['    + Interest Expense, Net']
                                             - is_tab.loc['Pretax Income (Loss), GAAP'])
    fs.loc['Profit before tax'] = is_tab.loc['Pretax Income (Loss), GAAP']
    fs.loc['Taxes'] = - is_tab.loc['  - Income Tax Expense (Benefit)']
    fs.loc['Other losses and minority interest'] = - (is_tab.loc['Pretax Income (Loss), GAAP']
                                                      - is_tab.loc['  - Income Tax Expense (Benefit)']
                                                      - is_tab.loc['Net Income, GAAP'])
    fs.loc['Profit after tax'] = is_tab.loc['Net Income, GAAP']
    fs.loc['Dividends'] = - is_tab.loc['Total Cash Common Dividends']
    fs.loc['Retained earnings'] = is_tab.loc['Net Income, GAAP'] - is_tab.loc['Total Cash Common Dividends']
    fs = fs.astype(float)
    return(fs)    
    
# -----------------------------------------------------------
# [1] Enterprise valuation accounting approach

def enterpriseValueAccounting(firmBalanceSheet, year):
    # Left Hand Side
    netWorkingCapital = firmBalanceSheet.loc['Current assets'] - firmBalanceSheet.loc['Current liabilities']
    longTermAsset = firmBalanceSheet.loc['Net fixed assets'] + firmBalanceSheet.loc['Long-term investments'] + firmBalanceSheet.loc['Other long-term assets']
    enterpriseValue = netWorkingCapital + longTermAsset
    enterpriseValue = enterpriseValue[year]
    return enterpriseValue
# -----------------------------------------------------------
# [2] Enterprise valuation efficient market approach
def enterpriseValueEfficientMarket(ticker, firmBalanceSheet, year):
    netDebt = firmBalanceSheet.loc['Debt'] - firmBalanceSheet.loc['Cash and marketable securities']
    longTermLiability = firmBalanceSheet.loc['Other long-term liabilities']
    preferredEquity = firmBalanceSheet.loc['Preferred equity']
    minorityInterest = firmBalanceSheet.loc['Minority interest']
    # get the market value of common shares
    firmQuote = pdr.get_quote_yahoo(ticker)
    firmMarketCap = firmQuote.loc[ticker, 'marketCap']/1000000
    enterpriseValueMarket = netDebt + longTermLiability + preferredEquity + minorityInterest + firmMarketCap
    enterpriseValueMarket = enterpriseValueMarket[year]
    return enterpriseValueMarket

# -----------------------------------------------------------
# [3] rD: Average cost of existing debt


def averageCostOfDebt(firmBalanceSheet, firmIncomeStatement, year):
    netInterestPayment = -(firmIncomeStatement.loc['Interest payments on debt'] + firmIncomeStatement.loc['Interest earned on cash and marketable securities'])
    netDebt = firmBalanceSheet.loc['Debt'] - firmBalanceSheet.loc['Cash and marketable securities']
    averageNetDebt = (netDebt+netDebt.shift(1))/2
    costOfDebt = netInterestPayment/averageNetDebt
    costOfDebt = costOfDebt[year]
    return costOfDebt
    

# -----------------------------------------------------------
# [4] rD: Cost of debt based on rating-adjusted yield curve

# step 1 get the data for all bonds with the same rating with MRK


def costOfDebtYieldCurve(firmBond, sameRatedBonds):
    # step 2 for all bonds, get the years to maturity and yield to maturity
    sameRatedBonds['YTM']=sameRatedBonds['Yld to Mty (Mid)'].astype(float)/100
    sameRatedBonds['D_Mty'] = pd.to_datetime(sameRatedBonds['Maturity']) - pd.to_datetime('2021-11-28')
    sameRatedBonds['Y_Mty'] = sameRatedBonds['D_Mty']/np.timedelta64(1,'Y')


    # step 3 fit a polynomial model of yield to maturity on the years to maturity
    # step 3.1 remove the outliers using winsorize
    sameRatedBonds['YTM'] = sameRatedBonds['YTM'].clip(lower=sameRatedBonds['YTM'].quantile(0.01),
                                                       upper=sameRatedBonds['YTM'].quantile(0.99))
    sameRatedBonds['Y_Mty']=sameRatedBonds['Y_Mty'].clip(lower = sameRatedBonds['Y_Mty'].quantile(0.01),
                                                         upper = sameRatedBonds['Y_Mty'].quantile(0.99))
    # step 3.2 run polynomial model
    sameRatedBonds['Y_Mty^2'] = sameRatedBonds['Y_Mty']**2
    sameRatedBonds['Y_Mty^3'] = sameRatedBonds['Y_Mty']**3
    polyreg = LinearRegression().fit(X=sameRatedBonds[['Y_Mty', 'Y_Mty^2', 'Y_Mty^3']],y=sameRatedBonds['YTM'])
    r_square = polyreg.score(X=sameRatedBonds[['Y_Mty', 'Y_Mty^2', 'Y_Mty^3']],y=sameRatedBonds['YTM'])
    intercept = polyreg.intercept_
    coefs = polyreg.coef_

    # step 4 find the average years to maturity for MRK
    firmBond['D_Mty']=pd.to_datetime(firmBond['Maturity'])-pd.to_datetime('2021-11-28')
    firmBond['Y_Mty'] = firmBond['D_Mty']/np.timedelta64(1,'Y')
    y_mty = firmBond['Y_Mty'].mean()


    # step 5 predict the average yield to maturity for MRK using MRK's average
    # years to maturity and thepolynomial regression coefficients
    costOfDebt = intercept+coefs[0]*y_mty + coefs[1]*y_mty**2 + coefs[2]*y_mty**3
    out = {'Cost of Debt':costOfDebt, 'R-square':r_square}
    return out



# -----------------------------------------------------------
# [5] rE: Gordon dividend model

def costOfEquityGordon(ticker, evaluationDate, estimateWindow):
    # step 1 get the price
    firmQuote = pdr.get_quote_yahoo(ticker)
    firmPrice = firmQuote.loc[ticker,'price']
    
    #step 2 download the dividend payment
    firm_actions = pdr.get_data_yahoo_actions(ticker, start='1990-01-01', end= str(evaluationDate))
    #'action' is a column. only want 'DIVIDEND', remove 'SPLIT'
    firm_div = firm_actions.loc[firm_actions['action'] == 'DIVIDEND', :]
    
    #step 3 get the current divident payment
    firm_div.index = firm_div.index.to_period('Q')
    yearQnow = firm_div.index.max()
    divNowQuarter = firm_div.loc[yearQnow, 'value']
    
    #step 4 get the  dividend payments N years before
    n = estimateWindow
    yearQPast = yearQnow - n*4
    divPastQuarter = firm_div.loc[yearQPast, 'value']
    #step 5 calculate the quarterly, then annual dividend growth rate
    quarterlyGrowth = npf.rate(n*4,0,divPastQuarter,-divNowQuarter)
    annualGrowth = (1+quarterlyGrowth)**4 - 1
    #step annualize the current dividend
    divNowAnnual = divNowQuarter * 4 
    
    #step 7 cost of equity
    costOfEquity = divNowAnnual * (1+annualGrowth)/firmPrice + annualGrowth
    out = {'Cost of Equity': costOfEquity, 'Estimate Window': f'{yearQPast} to {yearQnow}', 'Dividend Growth': annualGrowth}
    return out



# -----------------------------------------------------------
# [6] rE: CAPM model

def CAPM(ticker, evaluationDate, estimateWindwo):
    evaluationDate = (pd.to_datetime(evaluationDate)).to_period('M')
    #estiemateWindow = how many years. So it should multiply by 12 = how many months
    startDate = evaluationDate - estimateWindwo * 12
    prices = pdr.get_data_yahoo([ticker, '^GSPC'], start=str(startDate), end=str(evaluationDate), interval='m')
    prices = prices['Adj Close']
    returns = prices/prices.shift(1) - 1
    returns = returns.dropna()
    regression = LinearRegression().fit(X=returns[['^GSPC']], y=returns[ticker])
    beta = float(regression.coef_)
    rSquare = regression.score(X=returns[['^GSPC']], y=returns[ticker])
    
    marketIndexStartDate = evaluationDate - 30*12
    marketPrice30Year = pdr.get_data_yahoo('^GSPC', start=str(marketIndexStartDate), end=str(evaluationDate), interval='m')
    marketPrice30Year = marketPrice30Year['Adj Close']
    marketReturn30Year = marketPrice30Year/marketPrice30Year.shift(1) - 1
    marketReturn30Year = marketReturn30Year.dropna()
    averageMarketReturn = marketReturn30Year.mean()
    annualMarketReturn = averageMarketReturn * 12
    riskFree30Year = pdr.DataReader('TB3MS', 'fred', start=str(marketIndexStartDate), end=str(evaluationDate))
    riskFreeRate = riskFree30Year['TB3MS'].mean()/100
    
    capm = riskFreeRate + beta * (annualMarketReturn - riskFreeRate)
    
    out = {'Cost of Equity': capm, 'CAPM beta': beta, 'R-square': rSquare}
    return out


# -----------------------------------------------------------
# [7] WACC weighted average cost of capital

def WACC(ticker, firmBalanceSheet, firmIncomeStatement, costOfEquity, costOfDebt, year):
    firmQuote = pdr.get_quote_yahoo(ticker)
    equity = firmQuote.loc[ticker, 'marketCap']/1000000
    debt = firmBalanceSheet.loc['Debt', year] - firmBalanceSheet.loc['Cash and marketable securities', year]
    taxRate = - firmIncomeStatement.loc['Taxes']/firmIncomeStatement.loc['Profit before tax']
    taxRate = taxRate.mean()
    wacc = equity/(equity+debt) * costOfEquity + debt/(equity+debt) * costOfDebt * (1-taxRate)
    out = {'WACC': wacc, 'equity weight': equity/(equity+debt), 'debt weight': debt/(equity+debt), 'tax rate': taxRate}
    return out

# -----------------------------------------------------------
# [8] EV: DCF approach based on CSCF



def CSCF(ticker, firmBalanceSheet, firmIncomeStatement, firm_cscf, wacc, shortGrowth, longGrowth, year):
    operatingCashFlow = firm_cscf.loc['Cash from Operating Activities']
    investingCashFlow = firm_cscf.loc['  + Change in Fixed & Intang']
    fcfBeforeInterest = operatingCashFlow + investingCashFlow
    
    taxRate = - firmIncomeStatement.loc['Taxes']/firmIncomeStatement.loc['Profit before tax']
    netInterestPayment = -firmIncomeStatement.loc['Interest payments on debt'] - firmIncomeStatement.loc['Interest earned on cash and marketable securities']
    InterestPaymentAdj = (1-taxRate) * netInterestPayment
    fcf = fcfBeforeInterest + InterestPaymentAdj
    
    fcfNow = fcf[year]
    
    cfForecast = pd.DataFrame(0,index=['fcf','terminalV','total'], columns = range(0,6))
    cfForecast.loc['fcf'] = npf.fv(shortGrowth, cfForecast.columns, 0, -fcfNow)
    cfForecast.loc['fcf', 0] = 0
    cfForecast.loc['terminalV', 5] = cfForecast.loc['fcf',5]*(1+longGrowth)/(wacc-longGrowth)
    cfForecast.loc['total'] = cfForecast.loc['fcf'] + cfForecast.loc['terminalV']
    enterpriseValue = npf.npv(wacc, cfForecast.loc['total'])*(1+wacc)**0.5

    cash = firmBalanceSheet.loc['Cash and marketable securities']
    financialLiability = firmBalanceSheet.loc['Debt'] + firmBalanceSheet.loc['Other long-term liabilities'] + firmBalanceSheet.loc['Preferred equity'] + firmBalanceSheet.loc['Minority interest']
    estimatedEquity = enterpriseValue + cash[year] - financialLiability[year]
    
    firmQuote = pdr.get_quote_yahoo(ticker)
    shares = firmQuote.loc[ticker, 'sharesOutstanding']/1000000
    estimatedPricePerShare = estimatedEquity/shares
    marketPricePerShare = firmQuote.loc[ticker, 'price']
    
    out = {'EV': enterpriseValue, 'Equity Value': estimatedEquity, 'Per Share Value': estimatedPricePerShare, 'Actual Price Per Share': marketPricePerShare, 'Future Cash Flows': cfForecast}
    return out
    
    
    
#%% Main code
# -----------------------------------------------------------
# Model input
import datetime

dataPath = '/Users/eric/Library/Containers/com.microsoft.Excel/Data/Desktop/Fordham/Financial Modeling/MRK(3).xlsx'
ticker = 'MRK'
year = 'FY 2019'
evaluationDate = datetime.date.today()

firmBalanceSheet = pd.read_excel(dataPath, ticker+'_BS')
firmBalanceSheet = rewrite_bs(tab_clean(firmBalanceSheet))

firmIncomeStatement = pd.read_excel(dataPath, ticker+'_IS')
firmIncomeStatement = rewrite_is(tab_clean(firmIncomeStatement))
firm_cscf = pd.read_excel(dataPath, ticker+'_CSCF')
firm_cscf = tab_clean(firm_cscf).dropna()

firmBond = pd.read_excel(dataPath, ticker+'_bond')
sameRatedBonds = pd.read_excel(dataPath, 'Same_rated_bonds')
sameRatedBonds = (sameRatedBonds[sameRatedBonds != '#N/A Field Not Applicable']).dropna()



# -----------------------------------------------------------
# Build models


firm_ev_acc = enterpriseValueAccounting(firmBalanceSheet, year)
firm_ev_mkt = enterpriseValueEfficientMarket(ticker, firmBalanceSheet, year)
firm_rD1 = averageCostOfDebt(firmBalanceSheet, firmIncomeStatement, year)
firm_rD2 = costOfDebtYieldCurve(firmBond, sameRatedBonds)
firm_rE1 = costOfEquityGordon(ticker, evaluationDate, 5)
firm_rE2 = CAPM(ticker, evaluationDate, 10)
firm_wacc = WACC(ticker, firmBalanceSheet, firmIncomeStatement, (firm_rE1['Cost of Equity']+firm_rE2['Cost of Equity'])/2, (firm_rD1+firm_rD2['Cost of Debt'])/2, year)
firm_ev_dcf = CSCF(ticker, firmBalanceSheet, firmIncomeStatement, firm_cscf, firm_wacc['WACC'], 0.06, 0.02, year)
print(f'''Firm: {ticker}, Date: {evaluationDate}
EV accounting approach: {firm_ev_acc} M
EV efficient market approach: {round(firm_ev_mkt,1)} M
rD average cost of existing debt: {round((firm_rD1*100),2)}%
rD based on rating-adjusted yield curve: {round((firm_rD2['Cost of Debt']*100),2)}%
rE based on gordon dividend model: {round((firm_rE1['Cost of Equity']*100),2)}%
rE CAPM: {round((firm_rE2['Cost of Equity']*100),2)}%
WACC: {round((firm_wacc['WACC']*100),2)}%
EV DCF approach: {round(firm_ev_dcf['EV'],1)} M
Price per share estimated: {round(firm_ev_dcf['Per Share Value'],2)}
''')


