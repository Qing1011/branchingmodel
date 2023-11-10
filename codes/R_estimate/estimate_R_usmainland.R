in_usmain = read.csv(file.choose())
library('EpiEstim')

numfips = ncol(in_usmain)-1
len_T = nrow(in_usmain)

df_R_median <- data.frame(matrix(ncol = numfips, 
                                 nrow = len_T-7))
df_R_005 <- data.frame(matrix(ncol = numfips, 
                              nrow = len_T-7))
df_R_095 <- data.frame(matrix(ncol = numfips, 
                              nrow = len_T-7))

column_names <- names(in_usmain)
colnames(df_R_median) <- column_names[2:(numfips+1)]
colnames(df_R_005) <- column_names[2:(numfips+1)]
colnames(df_R_095) <- column_names[2:(numfips+1)]

for (i in 2:3109){
  print(i)
  col_name = column_names[i]
  colnames(in_usmain)[colnames(in_usmain) == "col_name"] <- "I"
  res <- estimate_R(in_usmain[[i]], method = "parametric_si",
                          config = make_config(list(mean_si = 4.7, std_si = 2.9)))
  temp <- res$R
  df_R_median[[col_name]] <- temp$`Median(R)`
  df_R_005[[col_name]] <- temp$`Quantile.0.05(R)`
  df_R_095[[col_name]] <- temp$`Quantile.0.95(R)`
  }


write.csv(df_R_median,'EpiEstim_R_median.csv',row.names = FALSE)
write.csv(df_R_005,'EpiEstim_R_005.csv',row.names = FALSE)
write.csv(df_R_095,'EpiEstim_R_095.csv',row.names = FALSE)
