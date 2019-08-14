from markowitz_pkg.model.markowitz_model import *
from markowitz_pkg.preprocess.data_process import *


args = Namespace(
    file = 'markowitz_pkg/data/stock_price.xlsx',
    portion = 0.8,
    num_portfolios = 500
)


if __name__ == '__main__':
    dataset = Dataset(args.file, args.portion)

    markowitz = Markowitz_Portfolio(dataset)

    results, portfolios_weights = markowitz.generate_random_portfolios(args.num_portfolios)
    markowitz.find_min_risk_portfolio()
    markowitz.plot_efficient_curve()
    print(markowitz.optimal_returns)
    print(markowitz.optimal_risks)
