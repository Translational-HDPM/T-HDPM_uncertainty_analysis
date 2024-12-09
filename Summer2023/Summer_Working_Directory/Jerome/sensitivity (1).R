# sensitivity.R
#
# Generate the file with sensitivity and specificity for 
# each cutoff point.
#
# We assume that the cutoff point is closed to the right, 
# so that:
#
#   value <= cutoff   NCI
#   value > cutoff    ADI

################################################################################
# CONFIGURE

# Load handy packages.

library(tidyverse)

################################################################################
# INITIALIZE

# Read the data that was created previously.

Data <- read_csv("Transformed AD Dataset Feb 23 for Purdue.csv")

# Set data conventions.

Data <- Data %>%
  filter(Disease %in% c("AD", "NCI"))

################################################################################
# PROCESS

# Calculate the scores for each subject.

Scores <- Data %>%
  group_by(Disease, `Isolate ID`) %>%
  summarise(Score = sum(z * Coeff), .groups = "drop")

#Scores %>% ggplot(aes(x = Score, fill = Disease)) + geom_histogram(bins = 30, alpha = 0.5)

# Order the subjects by score.  Then calculate the sensitivity and specificity at
# each attained value.  Note that the sensitivity and specificity will not change
# until the next score value is attained.

Scores <- Scores %>%
  arrange(Score) 

n <- nrow(Scores)

Scores <- Scores %>% 
  mutate(
    Cutoff = Score,
    Sensitivity = NA_real_,
    Specificity = NA_real_
  )

n_AD <- sum(Scores$Disease == "AD")
n_NCI <- sum(Scores$Disease == "NCI")

for (i in 1:(n - 1)) {
  
  Scores$Sensitivity[i] <- sum(Scores$Disease[(i + 1):n] == "AD") / n_AD
  Scores$Specificity[i] <- sum(Scores$Disease[1:i] == "NCI") / n_NCI

}

# Fill in the endpoints.

Scores$Sensitivity[n] <- 0
Scores$Specificity[n] <- 1

################################################################################
# WRAP-UP

Scores %>% write_csv("Cutoffs.csv")

