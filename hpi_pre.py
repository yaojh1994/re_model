#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calendar import monthrange
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression, LinearRegression
from datetime import datetime

from zigzag import eff
plt.style.use('ggplot')

class Hpi_pre(object):

    def __init__(self):
        self.y = pd.read_csv('hpi.csv', index_col = 0, parse_dates = True)
        self.lny, self.lny_trend = self.cal_lny_trend()
        self.gap = self.y-self.lny_trend
        self.pre_processing()
        self.x = self.load_indic(1)

    def pre_processing(self):
        #pivots = eff(self.gap,0.03)
        #plt.plot(self.gap)
        #plt.plot(pivots[0],pivots[1])
        #print pivots
        self.gap['p'] = 0
        self.gap.loc['2007-06':'2007-11','p'] = 1
        self.gap.loc['2010-09':'2011-02','p'] = 1
        self.gap.loc['2014-01':'2014-05','p'] = 1

    def load_indic(self, cal_indic = False):
        if cal_indic:
            irl = pd.read_csv('irl.csv', index_col = 0, parse_dates = True)
            irs = pd.read_csv('irs.csv', index_col = 0, parse_dates = True)
            sratio = pd.read_csv('sratio.csv', index_col = 0, parse_dates = True)
            sz = pd.read_csv('sz.csv', index_col = 0, parse_dates = True)
            unr = pd.read_csv('unr.csv', index_col = 0, parse_dates = True)
            land_supply = pd.read_csv('land_supply.csv', index_col = 0, parse_dates = True)
            rei = pd.read_csv('rei.csv', index_col = 0, parse_dates = True)
            m1 = pd.read_csv('m1.csv', index_col = 0, parse_dates = True)
            m2 = pd.read_csv('m2.csv', index_col = 0, parse_dates = True)
            gdp = pd.read_csv('gdp.csv', index_col = 0, parse_dates = True)
            cpi = pd.read_csv('cpi.csv', index_col = 0, parse_dates = True)
            cpi_yoy = pd.read_csv('cpi_yoy.csv', index_col = 0, parse_dates = True)
            fr = pd.read_csv('fr.csv', index_col = 0, parse_dates = True)
            tp = pd.read_csv('tp.csv', index_col = 0, parse_dates = True)
            gini = pd.read_csv('gini.csv', index_col = 0, parse_dates = True)
            m1_gdp = pd.read_csv('m1_gdp.csv', index_col = 0, parse_dates = True)

            irl = irl.reindex(self.y.index).fillna(method='pad').dropna()
            irs = irs.reindex(self.y.index).fillna(method='pad').dropna()
            sratio = sratio.reindex(self.y.index).fillna(method='pad').dropna()/100
            sz = sz.pct_change().reindex(self.y.index).fillna(method='pad').dropna()
            unr = unr.reindex(self.y.index).fillna(method='pad').dropna()

            #land_supply = land_supply.reindex(self.y.index).fillna(method='pad').fillna(0)
            land_supply = land_supply.groupby(land_supply.index.strftime('%Y-%m')).sum()
            land_supply.index = [datetime(int(x[:4]),int(x[-2:]),monthrange(int(x[:4]),int(x[-2:]))[1]) for x in land_supply.index]
            land_supply = land_supply.rolling(6).mean().dropna()
            land_supply = land_supply.reindex(self.y.index).fillna(0)

            rei = rei.reindex(self.y.index).fillna(method='pad').dropna()
            m1 = m1.reindex(self.y.index).fillna(method='pad').dropna()
            m2 = m2.reindex(self.y.index).fillna(method='pad').dropna()
            gdp = gdp.pct_change().reindex(self.y.index).fillna(method='pad').dropna()
            cpi = cpi.reindex(self.y.index).fillna(method='pad').dropna()
            cpi_yoy = cpi_yoy.reindex(self.y.index).fillna(method='pad').dropna()
            fr = fr.reindex(self.y.index).fillna(method='pad').dropna()
            gini = gini.resample('m').last().fillna(method = 'pad')
            gini = gini.reindex(self.y.index).fillna(method='pad').dropna()
            tp = tp.pct_change().resample('m').last().fillna(method='pad')
            tp = tp.reindex(self.y.index).fillna(method='pad')
            m1_gdp = m1_gdp.rolling(5).apply(lambda x:(x[-1]-x[0])/x[0])
            m1_gdp = m1_gdp.reindex(self.y.index).fillna(method='pad')

            x = pd.concat([irl,irs,sratio,sz,unr,land_supply,rei,m1,m2,gdp,cpi,cpi_yoy,fr,gini,tp],1)
            x['m1_gdp'] = m1_gdp
            x['gap'] = self.gap['hpi']
            x['lnhpi'] = self.lny
            x['lnhpi_diff'] = self.lny.diff(2).fillna(0)
            x['lnhpi_mean'] = x['lnhpi_diff'].rolling(2).mean().fillna(0)
            x['lnhpi_trend'] = self.lny_trend
            x['irl_inv'] = 1/x['irl']
            x['irs_inv'] = 1/x['irs']
            x.to_csv('indic.csv', index_label = 'date')

        else:
            x = pd.read_csv('indic.csv', index_col = 0, parse_dates = True)
        return x

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        indic1 = 'gap'
        indic2 = 'm1_gdp'
        ax1.plot(self.x.loc[:, [indic1]], color = 'b', label = indic1)
        ax2.plot(self.x.loc[:, [indic2]], color = 'r', label = indic2)
        ax1.legend(loc = 2)
        ax2.legend(loc = 1)

        plt.show()


    def cal_lny_trend(self):
        lny = np.log(self.y)
        lny_train = lny[:'2014']
        lny_test = lny['2015':]
        x = np.arange(len(lny_train.hpi))
        y = np.array(lny_train.hpi)
        slope, intercept ,_,_,_ = linregress(x,y)
        lny_trend_train = slope*x+intercept

        lny_trend_test = []
        for date in lny_test.index:
            x = np.arange(len(lny[:date].hpi))
            y = np.array(lny[:date].hpi)
            slope, intercept ,_,_,_ = linregress(x,y)
            lny_trend_test.append((slope*x+intercept)[-1])
        lny_trend_test = np.array(lny_trend_test)
        lny_trend = np.append(lny_trend_train, lny_trend_test)

        lny_trend = pd.DataFrame(lny_trend, columns = ['hpi'], index = self.y.index)
        lny.columns = ['hpi']
        return [lny, lny_trend]

    def training(self):
        train_slice = slice('2005','2014')
        #x_train = self.x.loc[train_slice, ['gap']]
        x = self.x.loc[:,['irl_inv', 'm1', 'm1_gdp']]
        print x
        y = self.gap.loc[:, ['hpi']]
        x_train = x.loc[train_slice]
        y = y.shift(-12).fillna(method = 'pad')
        y_train = y.loc[train_slice]
        #lr = LogisticRegression()
        lr = LinearRegression()
        model = lr.fit(x_train, y_train)
        #result = model.predict_proba(x)[:,1]
        result = model.predict(x)
        result_df = pd.DataFrame(result, columns = ['prob'], index = y.index)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        #ax1.plot(self.x.loc[:, ['gap']], color = 'b')
        #ax2.bar(result_df[:'2014'].index, result_df[:'2014'].values.ravel(), width = 20, alpha = 0.5, color = 'y')
        #ax2.bar(result_df['2015':].index, result_df['2015':].values.ravel(), width = 20, alpha = 0.5, color = 'r')
        ax1.plot(y, color = 'b')
        ax2.plot(result_df[:'2014'].index, result_df[:'2014'].values.ravel(), color = 'y', linewidth = 2)
        ax2.plot(result_df['2014-11-30':].index, result_df['2014-11-30':].values.ravel(), color = 'r', linewidth = 2)
        plt.axvline('2016-08-31', color = 'k', linewidth = 1)

        plt.show()


if __name__ == '__main__':
    hpi_pre = Hpi_pre()
    hpi_pre.pre_processing()
    #hpi_pre.plot()
    hpi_pre.training()
