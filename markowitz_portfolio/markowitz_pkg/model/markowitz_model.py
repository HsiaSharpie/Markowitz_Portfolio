import sys
sys.path.append("../../")

from markowitz_pkg.preprocess.data_process import *
from collections import defaultdict
import cvxopt as opt
from cvxopt import blas, solvers
import math

class Markowitz_Portfolio(object):
    def __init__(self, dataset):
        '''
        Arguments:
            dataset(type): <class Dataset>
        '''
        self.origin_price_df = dataset.price_df
        self.origin_return_df = dataset.return_df
        self.origin_origin_mean_returns = np.asmatrix(self.origin_return_df.mean())
        self.origin_cov_matrix = np.asmatrix(self.origin_return_df.cov())


    def portfolio_performance(self, weights):
        '''
        Calculate return and risk for a portfolio and return it

        Return type:
            portfolio_return: list
            portfolio_risk: list
        '''
        # portfolio_return & risk are all annualized
        portfolio_return = np.asarray((weights * self.origin_origin_mean_returns.T)*252).flatten().tolist()
        portfolio_risk = np.asarray(((np.sqrt(weights * self.origin_cov_matrix * weights.T)))* np.sqrt(252)).flatten().tolist()

        return portfolio_return, portfolio_risk


    def generate_a_random_portfolio(self):
        '''
        Return return, risk, weights for a portfolio

        Return type:
            portfolio_return: list
            portfolio_risk: list
            weights: list
        '''
        num_assets = self.origin_return_df.shape[1] # number of assets
        weights = np.random.rand(num_assets)
        weights = np.asmatrix(weights / sum(weights)) # Sum of the weights need to be 1.

        # Calculate portfolio return, risk with method portfolio_performance
        portfolio_return, portfolio_risk = self.portfolio_performance(weights)

        # Convert the weights to ndarray and 1D
        weights = np.asarray(weights).flatten().tolist()

        return portfolio_return, portfolio_risk, weights


    def generate_random_portfolios(self, num_portfolios):
        '''
        Return results & weights for a portfolio

        Return type:
            results: defaultdict(list) with key (1)portfolio_return (2)portfolio_risk
            weights: list
        '''
        results = defaultdict(list)
        portfolios_weights = []

        for i in range(num_portfolios):
            portfolio_return, portfolio_risk, weights = self.generate_a_random_portfolio()

            portfolios_weights.append(weights)

            results['portfolio_return'].append(portfolio_return[0])
            results['portfolio_risk'].append(portfolio_risk[0])

        # portfolios_weights contain 500(num_portfolios) list of weights
        self.portfolios_weights = portfolios_weights
        # results contain two elements with first row: portfolio's return, second row: portfolio's_risk
        self.results = results

        return self.results, self.portfolios_weights

    def find_min_risk_portfolio(self):
        '''
        Through all of the random generated portfolios,
        we select the portfolio with minimum risk and plot it.

        Printout type:
            Annualized Return: float
            Annualized Risk: float
            min_vol_allocation: pandas.DataFrame

        Plot:
            Portfolio return & risk on the 2-D platform with x-axis: return, y-axis: risk
        '''
        min_risk_value = min(self.results['portfolio_risk'])

        for index, risk in enumerate(self.results['portfolio_risk']):
            if risk == min_risk_value:
                min_risk_index = index
            else:
                continue

        # Get the return & risk of the min_risk_portfolio
        min_risk_return = self.results['portfolio_return'][min_risk_index]
        min_risk_risk = self.results['portfolio_risk'][min_risk_index]

        # Get the weights corresponding to the min_risk_index
        min_vol_allocation = pd.DataFrame(self.portfolios_weights[min_risk_index],index=self.origin_return_df.columns,columns=['allocation'])
        min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
        min_vol_allocation = min_vol_allocation.T

        # Print out the annualized return & risk and weights of the minimum_risk portfolio
        print("Minimum Volatility Portfolio Allocation\n")
        print("Annualized Return:", round(min_risk_return,4))
        print("Annualized Risk:", round(min_risk_risk,4))
        print("\n")
        print(min_vol_allocation)

        # Plot all the portfolios and star the minimum_risk one
        plt.figure(figsize=(10, 7))
        plt.scatter(self.results['portfolio_risk'], self.results['portfolio_return'], marker='o', s=10, alpha=0.5)
        plt.scatter(min_risk_risk, min_risk_return,marker='*',color='g',s=80, label='Minimum Risk')
        plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
        plt.xlabel('annualized risks')
        plt.ylabel('annualized returns')
        plt.legend(labelspacing=0.8)
        plt.show()


    def plot_portfolios(self):
        '''
        Plot:
            Portfolio return & risk on the 2-D platform with x-axis: return, y-axis: risk
        '''
        plt.plot(self.results['portfolio_risk'], self.results['portfolio_return'], 'o', markersize=5)
        plt.xlabel('risk')
        plt.ylabel('return')
        plt.title('Return and risk of returns of randomly generated porfolios')
        plt.show()


    def optimal_portfolio(self):
        # Convert data to the form we need
        origin_returns = self.origin_return_df.values.T[:,1:]

        n = len(origin_returns)
        origin_returns = np.asmatrix(origin_returns)
        #
        N = 100
        mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
        #
        # Convert to cvxopt matrices
        S = opt.matrix(np.cov(origin_returns))
        pbar = opt.matrix(np.mean(origin_returns, axis=1))
        #
        # # Create constraint matrices
        G = -opt.matrix(np.eye(n))
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
        #
        # Calculate efficient frontier weights using quadratic programming
        portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]

        # Calcualte returns & risks
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

        # annualized the returns & risks
        self.optimal_returns = [return_*252 for return_ in returns]
        self.optimal_risks = [risk_*math.sqrt(252) for risk_ in risks]

        # Calculate the 2nd degree polynomial of the Frontier Curve
        m1 = np.polyfit(self.optimal_returns, self.optimal_risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])

        # Calculate the optimal portfolio weights
        self.optimal_weights = solvers.qp(opt.matrix(x1*S), -pbar, G, h, A, b)['x']


    def plot_efficient_curve(self):
        '''
        Plot:
            Portfolio return & risk on the 2-D platform with x-axis: return, y-axis: risk
        '''
        self.optimal_portfolio()

        # Plot the random generated portfolios
        plt.plot(self.results['portfolio_risk'], self.results['portfolio_return'], 'o', markersize=5)
        plt.xlabel('risk')
        plt.ylabel('return')
        plt.title('Return and risk of returns of randomly generated porfolios')

        # Plot the optimal portfolios
        plt.plot(self.optimal_risks, self.optimal_returns, 'y-o', markersize=5)
        plt.show()
