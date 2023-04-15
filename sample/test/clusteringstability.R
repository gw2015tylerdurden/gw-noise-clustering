setwd("~/workspace/gw-noise-clustering/R/data")

#必要なpackage
library(kernlab)

calinhara <- function(x,clustering,cn=max(clustering)){
  x <- as.matrix(x)
  p <- ncol(x)
  n <- nrow(x)
  cln <- rep(0,cn)
  W <- matrix(0,p,p)
  for (i in 1:cn)
    cln[i] <- sum(clustering==i)
  #  print(cln)
  for (i in 1:cn) {
    clx <- x[clustering==i,]
    cclx <- cov(as.matrix(clx))
    #    print(cclx)
    if (cln[i] < 2) 
      cclx <- 0 
    W <- W + ((cln[i] - 1) * cclx)
  }
  S <- (n - 1) * cov(x)
  B <- S - W
  out <- (n-cn)*sum(diag(B))/((cn-1)*sum(diag(W)))
  out
}

kmeansruns <- function(data,krange=2:10,criterion="ch",
                       iter.max=100,runs=100,
                       scaledata=FALSE,alpha=0.001,
                       critout=FALSE,plot=FALSE,...){
  data <- as.matrix(data)
  if (scaledata) data <- scale(data)
  if (criterion=="asw") sdata <- dist(data)
  cluster1 <- 1 %in% krange
  crit <- numeric(max(krange))
  km <- list()
  for (k in krange){
    if (k>1){
      minSS <- Inf
      kmopt <- NULL
      for (i in 1:runs){
        options(show.error.messages = FALSE)
        repeat{
          #          cat(k," ",i,"before \n")
          kmm <- try(kmeans(data,k,iter.max=iter.max,...))
          #          str(kmm)
          if (!inherits(kmm,"try-error")) break
          #         cat(k," ",i,"\n")
        }
        options(show.error.messages = TRUE)
        swss <- sum(kmm$withinss)
        #        print(calinhara(data,kmm$cluster))
        if (swss<minSS){
          kmopt <- kmm
          minSS <- swss
        }
        if (plot){
          par(ask=TRUE)
          pairs(data,col=kmm$cluster,main=swss)
        }
      } # for i
      km[[k]] <- kmopt
      #      print(km[[k]])
      #      print(calinhara(data,km[[k]]$cluster))
      crit[k] <- switch(criterion,
                        asw=cluster.stats(sdata,km[[k]]$cluster)$avg.silwidth,
                        ch=calinhara(data,km[[k]]$cluster))
      if (critout)
        cat(k," clusters ",crit[k],"\n")
    } # if k>1
  } # for k
  if (cluster1)
    cluster1 <- dudahart2(data,km[[2]]$cluster,alpha=alpha)$cluster1
  k.best <- which.max(crit)
  if (cluster1)
    k.best <- 1
  #  print(crit)
  #  print(k.best)
  #  print(km[[k.best]])
  km[[k.best]]$crit <- crit
  km[[k.best]]$bestk <- k.best
  out <- km[[k.best]]
  out 
}

kmeansCBI <- function(data,krange,k=NULL,scaling=FALSE,runs=1,criterion="ch",...){
  if (!is.null(k)) krange <- k
  if(!identical(scaling,FALSE))
    sdata <- scale(data,center=TRUE,scale=scaling)
  else
    sdata <- data
  c1 <- kmeansruns(sdata,krange,runs=runs,criterion=criterion,...)
  partition <- c1$cluster
  cl <- list()
  nc <- c1$bestk
  #  print(nc)
  #  print(sc1)
  for (i in 1:nc)
    cl[[i]] <- partition==i
  out <- list(result=c1,nc=nc,clusterlist=cl,partition=partition,
              clustermethod="kmeans")
  out
}

speccCBI <- function(data,k,...){
  #  require(kernlab)
  data <- as.matrix(data)
  options(show.error.messages = FALSE)
  c1 <- try(specc(data,centers=k,...))
  options(show.error.messages = TRUE)
  if (inherits(c1,"try-error")){
    partition <- rep(1,nrow(data))
    cat("Function specc returned an error, probably a one-point cluster.\n All observations were classified to cluster 1.\n")
  }
  else
    partition <- c1@.Data
  cl <- list()
  nc <- k
  #  print(nc)
  #  print(sc1)
  for (i in 1:nc)
    cl[[i]] <- partition==i
  out <- list(result=c1,nc=nc,clusterlist=cl,partition=partition,
              clustermethod="spectral")
  out
}

# New with "fn" and "lda" method
classifnp <- function (data, clustering, method = "centroid", 
                       cdist = NULL, centroids = NULL, nnk = 1){
  #  require(class)
  #  require(MASS)
  data <- as.matrix(data)
  k <- max(clustering)
  p <- ncol(data)
  n <- nrow(data)
  topredict <- clustering < 0
  
  ##### Average linkage 
  if (method == "averagedist") {
    if (is.null(cdist)) { cdist <- as.matrix(dist(data)) } 
    else                { cdist <- as.matrix(cdist) }
    
    prmatrix <- matrix(0, ncol = k, nrow = sum(topredict))
    #    print("classifnp")
    #    print(topredict)
    #    print(clustering)
    for (j in 1:k){
      prmatrix[, j] <- rowMeans(as.matrix(cdist[topredict, clustering == j,drop=FALSE]))
    }
    clpred <- apply(prmatrix, 1, which.min)
    clustering[topredict] <- clpred
  }
  
  #### Kmeans, PAM, specc, ...
  if (method == "centroid") {
    if (is.null(centroids)) {
      centroids <- matrix(0, ncol = p, nrow = k)
      for (j in 1:k) centroids[j, ] <- colMeans(as.matrix(data[clustering == j, ]))
    }
    #    print(centroids)
    #    print(sum(topredict))
    clustering[topredict] <- knn1(centroids, data[topredict,], 1:k)
  }
  
  #### Mclust
  if (method == "qda") {
    qq <- try(qda(data[!topredict, ], grouping = as.factor(clustering[!topredict])), silent = TRUE)
    
    if (identical(attr(qq, "class"), "try-error")) 
      qq <- lda(data[!topredict, ], grouping = as.factor(clustering[!topredict]))
    clustering[topredict] <- as.integer(predict(qq, data[topredict, ])$class)
    
  }
  if (method == "lda") {
    qq <- lda(data[!topredict, ], grouping = as.factor(clustering[!topredict]))
    clustering[topredict] <- as.integer(predict(qq, data[topredict, ])$class)
  }
  ### Single linkage, specClust
  if (method == "knn"){
    clustering[topredict] <- as.integer(knn(data[!topredict, ], 
                                            data[topredict, ], 
                                            as.factor(clustering[!topredict]), 
                                            k = nnk))
  } 
  
  #### Complete linkage
  if (method == "fn") {
    if (is.null(cdist)){ cdist <- as.matrix(dist(data)) } 
    else               { cdist <- as.matrix(cdist)      }
    
    fdist <- matrix(0, nrow=sum(topredict), ncol = k)
    for (i in 1:k) {
      fdist[,i] <- apply(as.matrix(cdist[topredict, clustering==i]), 1, max)
    }
    bestobs1 <- apply(fdist, 1, which.min)
    clustering[topredict] <- bestobs1
  }
  clustering
}


# Introduces parameter centroidname to indicate where in CBIoutput$result
# the centroids can be found (automatically for kmeansCBI, claraCBI);
# If NULL and data are not distances, mean is computed within classifnp
# largeisgood: Take 1-original value so that large is good (only stabk)
nselectboot <- function (data, B = 50, distances = inherits(data, "dist"),
                         clustermethod = NULL, 
                         classification = "averagedist", centroidname = NULL,
                         krange = 2:10, count = FALSE, 
                         nnk = 1, largeisgood=FALSE,...) 
{
  dista <- distances
  data <- as.matrix(data)
  if (classification == "averagedist") {
    if (dista) 
      dmat <- data
    else dmat <- as.matrix(dist(data))
  }
  stab <- matrix(0, nrow = B, ncol = max(krange))
  n <- nrow(data)
  for (k in krange) {
    if (count) 
      cat(k, " clusters\n")
    for (i in 1:B) {
      if (count) 
        print(i)
      #d1 <- sample(n, n, replace = TRUE)
      #d2 <- sample(n, n, replace = TRUE)
      # test
      d1 <- rep(seq(1, n, by=2), each=2)[1:n]
      d2 <- rep(seq(2, n, by=2), each=2)[1:n]
      d2[n] = n
      if (dista) {
        dmat1 <- data[d1, d1]
        dmat2 <- data[d2, d2]
      }
      else {
        dmat1 <- data[d1, ]
        dmat2 <- data[d2, ]
      }
      if (distances){
        if ("diss" %in% names(formals(clustermethod))){
          clm1 <- clustermethod(as.dist(dmat1), k = k, 
                                diss = TRUE, ...)
          clm2 <- clustermethod(as.dist(dmat2), k = k, 
                                diss = TRUE, ...)
        }
        else{
          clm1 <- clustermethod(as.dist(dmat1), k = k, ...)
          clm2 <- clustermethod(as.dist(dmat2), k = k, ...)
        }
      }
      else {
        #clm1 <- clustermethod(dmat1, k = k, ...)
        #clm2 <- clustermethod(dmat2, k = k, ...)
      }
      #            cl2 <- clm2$partition
      centroids1 <- centroids2 <- NULL
      cj1 <- cj2 <- rep(-1, n)
      #cj1[d1] <- clm1$partition
      #cj2[d2] <- clm2$partition
      # test
      cj1[d1] <- array(1:22, dim = c(n, 1))
      cj2[d2] <- array(1:22, dim = c(n, 1))
      #            centroids <- NULL
      if (classification == "centroid") {
        if (is.null(centroidname)){
          if (identical(clustermethod, kmeansCBI))
            centroidname <- "centers"
          if (identical(clustermethod, claraCBI))
            centroidname <- "medoids"
        }
        if (!is.null(centroidname)){
          centroids1 <- clm1$result[centroidname][[1]]
          centroids2 <- clm2$result[centroidname][[1]]
        }
      }
      #         print(centroidname)
      #           if (classification == "centroid" & dista){
      #           centroids1 <- unlist(centroids1)
      #           centroids2 <- unlist(centroids2,recursive=FALSE)
      #           }
      #          print(str(clm1))
      #          print(str(centroids1))
      if (dista) {
        #              print("classifdist")
        cj1 <- classifdist(data, cj1, method = classification, 
                           centroids = centroids1, nnk = nnk)
        #              print(cj1)
        #              print(cj2)
        #              print(centroids1)
        #              print(centroids2)
        cj2 <- classifdist(data, cj2, method = classification, 
                           centroids = centroids2, nnk = nnk)
      }
      else {
        cj1 <- classifnp(data, cj1, method = classification, 
                         centroids = centroids1, nnk = nnk)
        cj2 <- classifnp(data, cj2, method = classification, 
                         centroids = centroids2, nnk = nnk)
      }
      ctable <- table(cj1, cj2)
      #print(ctable)
      nck1 <- rowSums(ctable)
      stab[i, k] <- sum(nck1^2 - rowSums(ctable^2))
    }
  }
  stab <- stab/n^2
  stabk <- rep(NA, max(krange))
  for (k in krange) stabk[k] <- mean(stab[, k])
  kopt <- which.min(stabk)
  if (largeisgood){
    stabk <- 1-stabk
  }
  out <- list(kopt = kopt, stabk = stabk, stab = stab)
}

############################################################################
#真のラベル読み込み
#----------------------------------------------------
tmp <- read.csv("gravity_spy_labels.csv", header = FALSE)
lab_vec <- tmp[,2]
#----------------------------------------------------

#潜在空間のデータ読み込み
#----------------------------------------------------
tmp <- read.csv("z-autoencoder-outputs.csv", header = FALSE)
zdata <- tmp[,-1]
#----------------------------------------------------
############################################################################

#zdataのUMAPによる解析
############################################################################
#zdata_umap <- umap(zdata, n_components = 3, n_neighbors = 15, min_dist = 0.3, verbose = TRUE)
#n_components: 埋め込み次元数, min_dist: 広がり具合（小さくするとよりクラスタがまとまる）
zdata_umap <- read.csv("z_umap.csv", header = FALSE)
############################################################################
#ret = specc(as.matrix(zdata_umap), centers=22, kernel = "rbfdot", kpar = list(sigma = 1.0))
ret <- nselectboot(zdata_umap, B=1, clustermethod=speccCBI, krange=22:23, count=TRUE, kernel="rbfdot", kpar=list(sigma = 1.0))
print(ret)
