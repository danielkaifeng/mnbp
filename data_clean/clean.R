argv <- commandArgs(T)

dat <- read.table(argv[1], head=T, row.names=1, fill=T, sep=',')

dat[is.na(dat)] <- 0

write.csv(dat, argv[2], quote=F)
