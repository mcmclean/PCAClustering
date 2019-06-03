#!/usr/bin/env python3

'''
Small clustering framework

https://scikit-learn.org/stable/modules/clustering.html

note - revisit sklearn pipelines? might be neat
'''
# heavy-use
import numpy as np
import pandas as pd
# sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
# visualization
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LinearColorMapper, CategoricalColorMapper, Plot, Range1d, MultiLine, Circle
from bokeh.palettes import plasma, Spectral4
from bokeh.palettes import Category10 as palette
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.models.tools import BoxZoomTool, PanTool, WheelZoomTool, HoverTool, ResetTool
from bokeh.models.graphs import from_networkx


class PCAClusteringAnalysis:

	def __init__(self, in_df, ind_col, name_col):
		self.orig_data = in_df
		self.index_col = ind_col
		self.scaled_data = None
		self.pca_frame = None
		self.pca = None
		self.aug_data = None
		self.lookup = None
		self.name_col = name_col

	def missing_values_table(self):
		'''
		found this function online
		https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe
		'''
		self.lookup = self.orig_data[[self.index_col, self.name_col]].drop_duplicates()
		self.orig_data.drop([self.name_col], axis = 1, inplace = True)
		df = self.orig_data
		mis_val = df.isnull().sum()
		mis_val_percent = 100 * df.isnull().sum() / len(df)
		mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
		mis_val_table_ren_columns = mis_val_table.rename(
		columns = {0 : 'Missing Values', 1 : '% of Total Values'})
		mis_val_table_ren_columns = mis_val_table_ren_columns[
			mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
		'% of Total Values', ascending=False).round(1)
		print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
			"There are " + str(mis_val_table_ren_columns.shape[0]) +
			  " columns that have missing values.")
		print(mis_val_table_ren_columns)

	def scale_data(self):
		# perform PCA
		self.orig_data.set_index(self.index_col, inplace = True)
		scaler = StandardScaler()
		scaled_data = scaler.fit_transform(self.orig_data)
		scaled_data = pd.DataFrame(scaled_data, index = self.orig_data.index, columns = self.orig_data.columns)
		self.scaled_data = scaled_data

	def assess_multicollinearity(self, threshold = 0.8, visualize = True):
		c = self.scaled_data.corr().abs()
		s = c.unstack()
		so = s.sort_values(kind="quicksort")
		so = pd.DataFrame(so)
		so.reset_index(inplace = True)
		so.rename(inplace = True, columns = {'level_0': 'Feature 1', 'level_1': 'Feature 2', 0: 'Correlation'})
		so = so[so['Correlation'] >= threshold]
		so = so[so['Feature 1'] != so['Feature 2']]
		so.reset_index(inplace = True, drop = True)
		pairs = [(so.iloc[2*x]['Feature 1'], so.iloc[2*x+1]['Feature 1'], so.iloc[2*x]['Correlation']) for x in range(int(len(so)/2))]
		
		if visualize:
			''' Graph to examine feature correlation https://bokeh.pydata.org/en/latest/docs/user_guide/graph.html '''
			g = nx.Graph()
			for pair in pairs:
				g.add_edge(pair[0], pair[1], weight = pair[2])
			nx.draw(g, with_labels=True, font_size = 8)
			plt.show()

			# for start_node, end_node, _ in g.edges(data=True):
			# 	print(start_node)
			# 	print(end_node)
			# 	print(_)
			# 	print("\n")

			# # Show with Bokeh
			# plot = Plot(plot_width=400, plot_height=400, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
			# plot.title.text = "Graph Interaction Demonstration"

			# # node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("club", "@club")])
			# # plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

			# graph_renderer = from_networkx(g, nx.circular_layout, scale=1, center=(0, 0))

			# graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
			# graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)
			# plot.renderers.append(graph_renderer)

			# output_file("interactive_graphs.html")
			# show(plot)
			''' ------------------------------------ '''
		drop_features = [x for i, x in enumerate(list(so['Feature 1'].values)) if i % 2 == 0]
		cols = [x for x in self.scaled_data if x not in drop_features]
		self.scaled_data = self.scaled_data[cols]
		print(so)

	def fit_pca(self, num_components = 3):
		pca = PCA(n_components=num_components)
		pca_data = pca.fit_transform(self.scaled_data)
		df_pca = pd.DataFrame(pca_data, index = self.orig_data.index)
		self.pca_frame = df_pca
		self.pca = pca

	def print_pca_info(self):
		print("\nExplained variance ratio:  ")
		print(self.pca.explained_variance_ratio_)
		'''
		print("\nSingular Values:  ")
		print(self.pca.singular_values_)
		print("\nComponents:  ")
		print(self.pca.components_)
		'''
		print("\nTotal Explained Variance:  {0:0.2%}".format(np.sum(self.pca.explained_variance_ratio_)))

	def perform_clustering(self):
		# clustering = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit(self.pca_frame)

		# clustering = MiniBatchKMeans(n_clusters=3, random_state=0).fit(self.pca_frame) # current for suppliers
		clustering = MiniBatchKMeans(n_clusters=4, random_state=0).fit(self.pca_frame) # maybe better for suppliers???
		# clustering = MiniBatchKMeans(n_clusters=5, random_state=0).fit(self.pca_frame) # ???? looks like barf for suppliers

		# clustering = AgglomerativeClustering(n_clusters=3).fit(self.pca_frame)
		# clustering = DBSCAN(eps=.25, min_samples=2).fit(self.pca_frame)
		# labels = GaussianMixture(n_components = 3, random_state = 5).fit_predict(self.pca_frame) # department
		# n=3 interesting for departments ^, random state 1 or 2 or 4 or 5, though 5 looks like 1 or 2 I think?
		# labels = GaussianMixture(n_components = 2).fit_predict(self.pca_frame)

		try:
			labels = clustering.labels_.tolist()
		except NameError:
			print("Not a problem if using GMM")
		assert len(labels) == len(self.pca_frame), 'Label problem'
		self.pca_frame['clus_label'] = labels

	def visualize(self):
		# add axes titles
		'''
		# Seaborn viz
		ax = sns.scatterplot(x = 0, y = 1, hue = 'clus_label', data = self.pca_frame)
		plt.title('Example Plot')
		plt.xlabel('PCA1')
		plt.ylabel('PCA2')
		plt.show()
		'''
		list_x = list(self.pca_frame[0].values)
		list_y = list(self.pca_frame[1].values)
		names = self.lookup.set_index(self.index_col)
		names = names.reindex(index = self.pca_frame.index)
		names.reset_index(inplace = True)
		# desc = list(self.pca_frame.index.values)
		desc = list(names[self.name_col].values)
		labels = list(self.pca_frame['clus_label'].values)

		source = ColumnDataSource(data=dict(x=list_x, y=list_y, desc=desc, color=labels))
		hover = HoverTool(tooltips=[
			# ("index", "$index"),
			# ("(PCA1, PCA2)", "(@x, @y)"),
			(self.index_col, '@desc'),
		])
		zoom = BoxZoomTool()
		pan = PanTool()
		wheel = WheelZoomTool()
		mapper = LinearColorMapper(palette=plasma(256), low=min(labels), high=max(labels))
		# mapper = CategoricalColorMapper(palette=plasma(256), low=min(labels), high=max(labels))

		p = figure(plot_width=1000, plot_height=600, tools=[hover, zoom, pan, wheel], title="Clustering Test:  " + self.index_col)
		p.circle('x', 'y', size=10, source=source, color=transform('color', mapper)) # fill_color arg is okay but looks worse

		output_file('cluster_viz_' + self.index_col + '.html')
		show(p)

	def augment_original_data(self):
		# merge back in with original data
		col_rename = {col:'PC'+str(col+1) for col in list(self.pca_frame.columns) if type(col) == int}
		self.pca_frame.rename(inplace = True, columns = col_rename)
		self.pca_frame.reset_index(inplace = True)
		self.aug_data = pd.merge(self.orig_data.reset_index(), self.pca_frame, how = 'left', on = [self.index_col])

	def order_cols(self):
		first_comp = self.pca.components_[0]
		orig_cols = [x for x in list(self.orig_data.columns) if x is not self.index_col]
		cols = [x for x in list(self.scaled_data.columns) if x is not self.index_col]
		dropped = [x for x in orig_cols if x not in cols]
		assert len(first_comp) == len(cols)
		df = pd.DataFrame()
		df['col'] = pd.Series(cols)
		df['loading'] = pd.Series([x**2 for x in first_comp])
		df.sort_values(by = 'loading', inplace = True, ascending = False)
		df.reset_index(inplace = True, drop = True)
		print("\nLoadings:  ")
		print(df)
		rest = [x for x in list(self.aug_data.columns) if x not in cols]
		# self.aug_data = self.aug_data[list(df['col'].values) + rest]
		self.aug_data = self.aug_data[list(df['col'].values) + [x for x in rest if x not in dropped]]
		identifiers = self.aug_data[[self.index_col, 'clus_label']]
		identifiers = pd.merge(self.lookup, identifiers, how = 'inner', on = [self.index_col])
		identifiers.to_csv('mappings_' + self.index_col + '.csv', index = False)

	def get_cluster_means(self):
		means = self.aug_data.copy(deep = True)
		try:
			means.drop([self.index_col], axis = 1, inplace = True)
		except KeyError:
			pass
		cluster_ct = means.groupby('clus_label')['clus_label'].count()
		cluster_means = means.groupby('clus_label').mean()
		cluster_means.reset_index(inplace = True)
		cluster_means['Count'] = pd.Series(cluster_ct)
		round_cols = {x: 1 for x in list(cluster_means.columns) if x.count("PC") == 0 and x not in ['clus_label', 'Count']}
		cluster_means = cluster_means.round(round_cols)
		cluster_means.to_csv('cluster_means_' + self.index_col + '.csv', index = False)
		# count of each

	def clustering_main(self):
		self.missing_values_table()
		self.scale_data()
		self.assess_multicollinearity(visualize = False)
		self.fit_pca(num_components = 5)
		self.print_pca_info()
		self.perform_clustering()
		self.visualize()
		self.augment_original_data()
		self.order_cols()
		self.get_cluster_means()
