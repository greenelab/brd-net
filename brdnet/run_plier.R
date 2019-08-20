# This script runs PLIER on the data from download_categorized_data.ipynb

# CRAN dependencies
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

data.dir <- '../data'

args <- commandArgs(trailingOnly=TRUE)

if(length(args) < 2){
	print('As no value of k was passed in, it will be estimated from the data')
} else if (length(args) > 2){
	print('Usage: Rscript run_plier.R outdir [k]')
	quit(status=1)
} else if (length(args) == 0){
	print('Usage: Rscript run_plier.R outdir [k]')
	quit(status=1)
}

# Set the random seed
set.seed(42)

# Load and combine data from download_categorized_data.ipynb
disease.df <- read.table(file.path(data.dir, 'plier_disease.tsv'), sep = '\t', stringsAsFactors = FALSE, header = TRUE)
healthy.df <- read.table(file.path(data.dir, 'plier_healthy.tsv'), sep = '\t', stringsAsFactors = FALSE, header = TRUE)

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

k.estimate = NaN
out.dir <- args[1]
if (length(args) == 1){
	k.estimate <- PLIER::num.pc(row.normalized[common.genes, ])
} else if (length(args) == 2){
	k.estimate <- as.numeric(args[2])
}

plier.results <- PLIER(row.normalized[common.genes, ], allPaths, k=k.estimate)

# Save PLIER loadings
write.table(plier.results$Z, file=file.path(out.dir, paste('plier', k.estimate, 'Z.tsv', sep='_')), sep='\t', row.names=TRUE)
