library(dplyr) 
library(readr)
library(tidyr)
library(stringr) 
library(data.table)
library(parallel)
library(mltools)

setwd("/Users/ahn 1/Desktop")

myfread = function(x){
  print(x)
  Year = gsub(x, pattern = ".txt", replace = "")
  Data = fread(x, colClasses="")
  Data[, Year := Year]
  return(Data)
}

#need to read in since it's been manually changed
data_file = "/Users/ahn 1/Desktop/loan.txt"
data = fread(data_file, colClasses = "")
# ----------- START ------------

# Support file to map State names to two-letter abbreviations
state.mapping <- read.csv("/Users/ahn 1/Downloads/StateCodeMapping.csv", header = TRUE, stringsAsFactors = FALSE)

# -----------------------------------------------------------------------------------------------
# PART I: Read in raw data files, clean, and prep for joins
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
## A. Read and prep raw loan data file
dat.00 <- data
dat.10 <- dat.00 %>%     # dat.10 is the raw data, with a row number added as a unique record identifier
  mutate(Key = row_number())
head(dat.20$issue_d)
head(new.vars.00)
# Update issue_d and last_pymnt_d based on format in which they are read; apply standard formatting for key use
if (nchar(word(dat.10[1,]$issue_d,1,sep=fixed('-'))) == 3) {
  dat.20 <- dat.10 %>%     # dat.20 will be used to calculate new columns; then those will join back on dat.10
    select(Key, loan_status, issue_d, last_pymnt_d, addr_state) %>%
    filter(loan_status %in% c('Current', 'Charged Off', 'Fully Paid')) %>%
    # NOTE: last_pymnt_d_key is created to force any 'Current' loan to receive Feb 2019 data
    #       issue_d_key created to force standard formatting on single digit years
    mutate(issue_Mon = word(issue_d,1,sep=fixed('-')), 
           issue_Yr = ifelse(nchar(word(issue_d,2,sep=fixed("-"))) == 1, 
                             paste(0,word(issue_d,2,sep=fixed('-')),sep=""), 
                             word(issue_d,2,sep=fixed('-'))),
           last_Mon = word(last_pymnt_d,1,sep=fixed('-')),
           last_Yr = ifelse(nchar(word(last_pymnt_d,2,sep=fixed("-"))) == 1, 
                            paste(0,word(last_pymnt_d,2,sep=fixed('-')),sep=""), 
                            word(last_pymnt_d,2,sep=fixed('-'))),
           issue_d_key = paste(issue_Yr, issue_Mon, sep="-"),
           last_pymnt_d_key = ifelse(loan_status == 'Current', 
                                     '2019-Feb', 
                                     paste(last_Yr,last_Mon,sep="-"))) %>%
    select(-c(issue_Mon, issue_Yr, last_Mon, last_Yr))
} else {
  dat.20 <- dat.10 %>%     # dat.20 will be used to calculate new columns; then those will join back on dat.10
    select(Key, loan_status, issue_d, last_pymnt_d, addr_state) %>%
    filter(loan_status %in% c('Current', 'Charged Off', 'Fully Paid')) %>%
    # NOTE: last_pymnt_d_key is created to force any 'Current' loan to receive Feb 2019 data
    #       issue_d_key created to force standard formatting on single digit years
    mutate(last_pymnt_d_key = ifelse(loan_status == 'Current', '2019-Feb',
                                     ifelse(nchar(word(last_pymnt_d,1,sep=fixed("-"))) == 1, 
                                            paste(0,last_pymnt_d,sep=""), 
                                            last_pymnt_d)),
           issue_d_key = ifelse(nchar(word(issue_d,1,sep=fixed("-"))) == 1, 
                                paste(0,issue_d,sep=""), 
                                issue_d))
}

## B. Read and prep S&P 500 data
sp500.00 <- read.csv("/Users/ahn 1/Downloads/SP500_Yahoo.csv", header = TRUE, stringsAsFactors = FALSE)
sp500.10 <- sp500.00 %>%
  select(Date, Close) %>%
  mutate(mmm = ifelse(as.integer(substr(Date,1,nchar(Date)-7))-1 > 0, 
                      as.integer(substr(Date,1,nchar(Date)-7))-1, 
                      12), 
         Mon = month.abb[mmm],
         Yr = ifelse(Mon == 'Dec' & as.integer(word(Date,3,sep=fixed('/'))) < 2011,
                     paste(0,as.integer(substr(Date,nchar(Date)-1,nchar(Date)))-1,sep=""),
                     ifelse(Mon == 'Dec', 
                            as.integer(substr(Date,nchar(Date)-1,nchar(Date)))-1,
                            substr(Date,nchar(Date)-1,nchar(Date))))) %>%
  unite(NewDate, Yr, Mon, sep = "-") %>%
  select(Date = NewDate, Close)

head(sp500.10)

## C. Read and prep federal funds data
funds.00 <- read.csv("/Users/ahn 1/Downloads/FundRates_FRED.csv", header = TRUE, stringsAsFactors = FALSE)
funds.10 <- funds.00 %>%
  mutate(mmm = ifelse(as.integer(substr(DATE,1,nchar(DATE)-7))-1 > 0, 
                      as.integer(substr(DATE,1,nchar(DATE)-7))-1, 
                      12),
         Mon = month.abb[mmm],Yr = substr(DATE,nchar(DATE)-1,nchar(DATE))) %>%
  unite(NewDate, Yr, Mon, sep = "-") %>%
  select(Date = NewDate, Rate = EFFR)

## D. Read and prep state GDP data
### NOTE: Data for 2018 and 2019 is identical to 2017, which is the last available year
gdp.00 <- read.csv("/Users/ahn 1/Downloads/StateGDP_BEA.csv", header = TRUE, stringsAsFactors = FALSE)
gdp.10 <- gdp.00 %>%
  select(State = GeoName, X1997:X2019) %>%
  left_join(state.mapping, by = 'State') %>%
  select(-State) %>%
  select(State = State_Code, everything()) %>%
  filter(!is.na(State)) %>%
  group_by(State) %>%
  gather(key = 'Yr', value = 'GDP', X1997:X2019) %>%
  mutate(Year = substr(Yr,4,5)) %>%
  select(State, Year, GDP)

## E. Read and prep home price data
### NOTE: Data for Feb. 2019 is identical to that of Jan. 2019, which is the last available.
homes.00 <- read.csv("/Users/ahn 1/Downloads/HomeValuePerSqFt_ZILLOW.csv", header = TRUE, stringsAsFactors = FALSE)
homes.10 <- homes.00 %>%
  select(State = RegionName,everything()) %>%
  left_join(state.mapping, by = 'State') %>%
  select(State = State_Code, 4:277) %>%
  gather(key = 'Date', value = 'HomeValue', X1996.04:X2019.01) %>%
  mutate(mmm = as.integer(substr(Date,7,8)), Mon = month.abb[mmm], Yr = substr(Date,4,5)) %>%
  unite(NewDate, Yr, Mon, sep = "-") %>%
  select(State, Date = NewDate, HomeValue)

## F. Read and prep crime data
### NOTE: Crime rates for 2018 and 2019 equal that of 2017, which is the last available.
crimes.00 <- read.csv("/Users/ahn 1/Downloads/CrimeRates_STATISTICA.csv", header = TRUE, stringsAsFactors = FALSE)
crimes.10 <- crimes.00 %>%
  mutate(Yr = Year, Year = substr(Yr,3,4)) %>%
  select(Year, CrimeRate)

## G. Read and prep unemployment data
### NOTE: Data for Jan. and Feb. 2019 are identical to Dec. 2018, which is the last available
jobs.00 <- read.csv("/Users/ahn 1/Downloads/Unemployment_FRED.csv", header = TRUE, stringsAsFactors = FALSE)
jobs.10 <- jobs.00 %>%
  gather(key = 'State_Code', value = 'UnempRate', AKUR:WYUR) %>%
  mutate(State = substr(State_Code,1,2), mmm = as.integer(word(DATE,1,sep=fixed('/'))), Mon = month.abb[mmm], Yr = substr(word(DATE,3,sep=fixed('/')),3,4)) %>%
  unite(Date, Yr, Mon, sep = "-") %>%
  select(Date, State, UnempRate)

head(jo.10)
head(new.vars.00)
head(dat.20)

new.vars.00 <- dat.20 %>%
  mutate(issue_yr = substr(issue_d_key,3,4), end_yr = substr(last_pymnt_d_key,1,2)) 

# -----------------------------------------------------------------------------------------------
# PART II: Join economic indicators and calculate new columns
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
new.vars.00 <- dat.20 %>%
  mutate(issue_yr = substr(issue_d_key,1,2), end_yr = substr(last_pymnt_d_key,1,2)) %>%
  
  # S&P data
  merge(sp500.10, by.x = 'issue_d_key', by.y = 'Date', all.x = TRUE) %>%
  rename(Close_Start = Close) %>%
  merge(sp500.10, by.x = 'last_pymnt_d_key', by.y = 'Date', all.x = TRUE) %>%
  rename(Close_End = Close) %>%
  mutate(SP500_Change = round((Close_End - Close_Start) / Close_Start, 3)) %>%
  select(-c(Close_Start, Close_End)) %>%                                       
  # SP data done
  
  # Funds data
  merge(funds.10, by.x = 'issue_d_key', by.y = 'Date', all.x = TRUE) %>%
  rename(Rate_Start = Rate) %>%
  merge(funds.10, by.x = 'last_pymnt_d_key', by.y = 'Date', all.x = TRUE) %>%
  rename(Rate_End = Rate) %>%
  mutate(FedRate_Change = round((Rate_End - Rate_Start) / Rate_Start, 3)) %>%  
  select(-c(Rate_Start, Rate_End)) %>%                                         
  # Funds data done
  
  # GDP data
  merge(gdp.10, by.x = c('addr_state', 'issue_yr'), by.y = c('State', 'Year'), all.x = TRUE) %>%
  rename(GDP_Start = GDP) %>%
  merge(gdp.10, by.x = c('addr_state', 'end_yr'), by.y = c('State', 'Year'), all.x = TRUE) %>%
  rename(GDP_End = GDP) %>%
  mutate(GDP_Change = round((GDP_End - GDP_Start) / GDP_Start, 3)) %>%  
  select(-c(GDP_Start, GDP_End)) %>% 
  # GDP data done
  
  # Homes data
  merge(homes.10, by.x = c('addr_state', 'issue_d_key'), by.y = c('State', 'Date'), all.x = TRUE) %>%
  rename(HomeValue_Start = HomeValue) %>%
  merge(homes.10, by.x = c('addr_state', 'last_pymnt_d_key'), by.y = c('State', 'Date'), all.x = TRUE) %>%
  rename(HomeValue_End = HomeValue) %>%
  mutate(HomeValue_Change = round((HomeValue_End - HomeValue_Start) / HomeValue_Start, 3)) %>%  
  select(-c(HomeValue_Start, HomeValue_End)) %>%
  # Homes data done
  
  # Crimes data
  merge(crimes.10, by.x = 'issue_yr', by.y = 'Year', all.x = TRUE) %>%
  rename(CrimeRate_Start = CrimeRate) %>%
  merge(crimes.10, by.x = 'end_yr', by.y = 'Year', all.x = TRUE) %>%
  rename(CrimeRate_End = CrimeRate) %>%
  mutate(CrimeRate_Change = round((CrimeRate_End - CrimeRate_Start) / CrimeRate_Start, 3)) %>%
  select(-c(CrimeRate_Start, CrimeRate_End)) %>%
  # Crime data done
  
  # Jobs data
  merge(jobs.10, by.x = c('addr_state', 'issue_d_key'), by.y = c('State', 'Date'), all.x = TRUE) %>%
  rename(UnempRate_Start = UnempRate) %>%
  merge(jobs.10, by.x = c('addr_state', 'last_pymnt_d_key'), by.y = c('State', 'Date'), all.x = TRUE) %>%
  rename(UnempRate_End = UnempRate) %>%
  mutate(UnempRate_Change = round((UnempRate_End - UnempRate_Start) / UnempRate_Start, 3)) %>%  
  select(-c(UnempRate_Start, UnempRate_End))
  # Jobs data done

head(new.vars.00)
head(dat.10)


# -----------------------------------------------------------------------------------------------
# PART III: Join new columns on to original data set and create new .csv file
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
new_data <- dat.10 %>%
  left_join(select(new.vars.00, Key, SP500_Change, FedRate_Change, GDP_Change, HomeValue_Change,
                   CrimeRate_Change, UnempRate_Change), by = 'Key') %>%
  select(-Key)

head(new_data)

#######################
# CONTINOUS VARIABLES #
#######################
data[is.na(new_data)] <- 0
#if less than 10% are non zero drop column, and drop empty columns
df <- subset(new_data, select = c("loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate", "installment", "sub_grade", "annual_inc", "emp_length", "verification_status", "loan_status", "issue_d",
                              "pymnt_plan", "purpose", "earliest_cr_line", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "revol_bal", "revol_util", "total_acc","initial_list_status",
                              "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d","application_type", "tot_coll_amt",
                              "tot_cur_bal", "open_acc_6m", "open_act_il","open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util","total_rev_hi_lim",
                              "inq_fi", "total_cu_tl", "inq_last_12m","addr_state", "acc_now_delinq", "acc_open_past_24mths", 
                              "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc", "mths_since_recent_inq",
                              "avg_cur_bal", "bc_util", "bc_open_to_buy", "chargeoff_within_12_mths", "delinq_2yrs", "dti", "home_ownership", "mths_since_last_delinq",
                              "num_accts_ever_120_pd", "percent_bc_gt_75", "pct_tl_nvr_dlq", "pub_rec_bankruptcies",
                              "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit"))

#dealing with continous variables
df$term = as.numeric(gsub(" months", "", df$term))
df$emp_length = as.character(gsub("+ years", "", df$emp_length))
df$emp_length = as.character(gsub('[[:punct:]]', '', df$emp_length))
df$issue_d = substring(df$issue_d, 5,8)
df$earliest_cr_line = substring(df$earliest_cr_line, 5,8)
df$last_pymnt_d = substring(df$last_pymnt_d, 5,8)
df$last_credit_pull_d = substring(df$last_credit_pull_d, 5,8)

#dealing with missing entries, filling in with 0
df[df==""] <- "0"

#fwrite(df, "lending_club.txt", sep="\t", quote = FALSE)

######################
# SUMMARY STATISTICS #
######################
df[, loan_amnt := as.numeric(loan_amnt)]
df[, int_rate := as.numeric(int_rate)]
df[, annual_inc := as.numeric(annual_inc)]

##quantile distribution of loan size
summary(df[, loan_amnt]) 
#1st q 8000, median 12900, mean 15047, 3rd q 20000, max 45000

##quantile distribution of interest rate
summary(df[, int_rate])
#1st q interest rate 9.49%, median 12.62%, mean 13.09%, 3rd quartile 16%, max 31%

##quantile distribution of annual_inc
summary(df[, annual_inc])
#1st q 46000, median 65000, mean 77992, 3rd q 93000, max 110000000 
