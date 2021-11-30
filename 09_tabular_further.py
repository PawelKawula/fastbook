#!pip install -Uqq fastbook kaggle waterfallcharts treeinterpreter dtreeviz
import fastbook

from fastbook import *
from kaggle import api
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG

pd.options.display.max_rows = 20
pd.options.display.max_columns = 8

path = URLs.path('icecream-shop-analysis')

Path.BASE_PATH = path

if not path.exists():
    path.mkdir(parents=true)
    api.dataset_download_files('sunlight1/icecream-shop-analysis', path=path, unzip=True)

    path.ls(file_type='text')

dt_flv = pd.read_csv(path / "icecream_flavors.csv")
dt_flv = dt_flv.rename(columns={"week": "saleWeek"}, inplace=False)
dt_sales = pd.read_csv(path / "icecream_sales.csv")
dt_sales = dt_sales.rename(columns={"date": "sale"}, inplace=False)
dt_sales['sales'] = dt_sales['sales'].astype(float)
dt_sales = add_datepart(dt_sales, "sale")

dt_sales['saleElapsed'] /= 3600 * 24
dt_sales = dt_sales.astype({"saleElapsed": "int64", "sales": "float64"})
dt_sales.head()

procs = [Categorify, FillMissing]

dep_var = 'sales'
cond = (dt_sales.saleMonth < 10)
train_idx = np.where(cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx), list(valid_idx))

cont, cat = cont_cat_split(dt_sales, 1, dep_var=dep_var)

to = TabularPandas(dt_sales, procs, cat, cont,
                   y_names=dep_var, splits=splits)

save_pickle(path / 'to.pkl', to)

to = load_pickle(path / 'to.pkl');

xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y

m = DecisionTreeRegressor(min_samples_leaf=8)
m.fit(xs, y)


def r_mse(pred, y): return round(math.sqrt(((pred - y) ** 2).mean()), 6)


def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


m_rmse(m, xs, y)

samp_idx = np.random.permutation(len(y))[:500]

dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
         fontname='DejaVu Sans', scale=0.4, label_fontsize=10,
         orientation='LR')


# 2
class Branch:
    def __init__(self, col_type="", value=0, y=0):
        self.col_name = ""
        self.col_type = col_type
        self.value = value
        self.y = y
        self.left, self.right = None, None

    def attach_children(self, left, right):
        self.left = left
        self.right = right

    def predict(self, x):
        if self.left is None:
            return self.y
        if col_type == 'cont':
            if x < value:
                return self.left.predict(x)
            return self.right.predict(x)
        if x != value:
            return self.left.predict(x)
        return self.right.predict(x)


class MyTreeRegressor:
    def __init__(self, min_samples_leaf=8):
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, xs, y):
        self.tree = Branch(y=xs.mean())
        self._fit_helper(xs, y, self.tree)

    def _fit_helper(self, xs, y, branch):
        if len(xs) < self.min_samples_leaf:
            return
        cond, minLowerBatch, minUpperBatch, batchVal, minCol = None, None, None, None, None
        for cName in xs.columns:
            minUpperCol, bestVal, minLowerCol = None, None, None
            for val in xs[cName]:
                if cName in cont:
                    cond = xs[cName] < val
                else:
                    cond = xs[cName] != val
                lower = xs.loc[cond]
                upper = xs.loc[~cond]
                if len(upper) < self.min_samples_leaf or len(lower) < self.min_samples_leaf:
                    continue
                if minLowerCol is None or lower.std().mean() + upper.std().mean() < minLowerCol.std().mean() + minUpperCol.std().mean():
                    minLowerCol, minUpperCol, bestVal = lower, upper, val
            if batchVal is None or minLowerCol.std().mean() + minUpperCol.std().mean() < minLowerBatch.std().mean() + minUpperBatch.std().mean():
                minLowerBatch, minUpperBatch, batchVal = minLowerCol, minUpperCol, bestVal
                minCol = cName
        left, right = None, None
        if minCol in cont:
            cond = xs[minCol] < batchVal
            branch.col_type = 'cont'
            left = Branch(minCol, batchVal, y[cond].mean())
            right = Branch(minCol, batchVal, y[~cond].mean())
        else:
            cond = xs[minCol] == batchVal
            branch.col_type = 'cat'
            left = Branch(minCol, batchVal, y[cond].mean())
            right = Branch(minCol, batchVal, y[~cond].mean())
        self._fit_helper(xs[cond], y[cond], left)
        self._fit_helper(xs[~cond], y[~cond], right)
        branch.attach_children(left, right)
        branch.col_name = minCol
        print(minCol, le(xs))

    def predict(self, x):
        return self.tree(x)


mytree = MyTreeRegressor()
mytree.fit(xs[:8], y[:8])
