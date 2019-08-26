# This script runs PLIER on the data from download_categorized_data.ipynb

# CRAN dependencies
library('argparser')
library('dplyr')

# Bioconductor dependencies
library('biomaRt')

# Github dependencies
# Install PLIER if it isn't already present
if (!require('PLIER',character.only = TRUE)) {
	library('devtools')
	tryCatch({
		install_github('wgmao/PLIER')
	},
	error=function(e){
		# For some reason devtools struggles to find the tar installation, this should fix it
		Sys.setenv(TAR = "/bin/tar")
		install_github('wgmao/PLIER')
	})

}
library('PLIER')

parser <- arg_parser('Run PLIER on gene expression data')
parser <- add_argument(parser, 'plierHealthy',
					   help='Path to a dataframe containing healthy gene expression data', 
					   type='character')
parser <- add_argument(parser, 'plierDisease', 
					   help='Path to a dataframe containing unhealthy gene expression data', 
					   type='character')
parser <- add_argument(parser, 'outdir', help='The directory to print results to', type='character')
parser <- add_argument(parser, '-k', help='The number of PLIER PCs to use', 
					   type='numeric')
args <- parse_args(parser)

# Set the random seed
set.seed(42)

# Load and combine data from download_categorized_data.ipynb
disease.df <- read.table(args$plierDisease, sep = '\t', stringsAsFactors = FALSE, header = TRUE)
healthy.df <- read.table(args$plierHealthy, sep = '\t', stringsAsFactors = FALSE, header = TRUE)

# Combine the disease and healthy dataframes
all.df <- merge.data.frame(healthy.df, disease.df, by='row.names')

row.names(all.df) <- all.df['Row.names'][[1]]
all.df['Row.names'] <- NULL

data("canonicalPathways")
data("bloodCellMarkersIRISDMAP")
data("svmMarkers")
# Combine three biological pathway datasets
allPaths <- PLIER::combinePaths(bloodCellMarkersIRISDMAP, canonicalPathways, svmMarkers)

all.matrix <- as.matrix(all.df)
# This portion is based on the function PLIERNewData from  
# https://github.com/greenelab/multi-plier/blob/master/util/plier_util.R 
# Keep only genes found in both the id matrix and the prior knowlege gene sets
common.genes <- PLIER::commonRows(all.matrix, allPaths)
row.normalized <- PLIER::rowNorm(all.matrix)

k.estimate = NULL
out.dir <- args$outdir
if (is.null(args$k)){
	k.estimate <- PLIER::num.pc(row.normalized[common.genes, ])
} else{
	k.estimate <- args$k
}

plier.results <- PLIER(row.normalized[common.genes, ], allPaths, k=k.estimate)

# Save PLIER loadings
write.table(plier.results$Z, file=file.path(out.dir, paste('plier', k.estimate, 'Z.tsv', sep='_')), sep='\t', row.names=TRUE)
