# 2025-10-13
This is a first version using simulated data based upon the live balance panel on the CEU server. The panel version has some major flaws so this is just a proof of concept but the data can be updated anytime.

Simulation: simulate_from_params.ipynb
Dashboards: cross_section_dashboard.py and panel_dashboard.py (type `streamlit run panel_dashboard.py' to run)

# 2025-10-29

## TODO

### Histogram

    - Get distribution parameters for industries separately --> DONE
        - Add moments up to 5 --> DONE
    - Add winzorize button to dashboards (2% at top and bottom) --> DONE
    - Hist settings: no need to add n filter --> DONE

### Scatterplot
    - run regressions within industries --> DONE
    - user can choose industry, x and y, add a line (linear, quadratic, cubic, or stepwise function - 5 and 20) --> DONE
    - add winzorize here as well --> DONE
    - Have a compare part, where two scatterplots of two industries can be compared
    - it should show the function parameters (can be turned off)

### Multivariate regression
    - similar to already existing dashboard
    - y: sales growth (2 versions: 2021-2019 over 2019 and log2021-log2019)
    - user can choose X-s - as in book -> for now use variables that we already export
    - each variable can be added in log
    - 3 categories for emp and sales
    - 1 interaction: emp_cat * ownership
    - if everything is ready, do the same where y: exit between 2019 and 2021

#### Server things: 
    - by industry distributions --> DONE
    - by industry models --> DONE

#### Design things:
    - show name of industries next to numbers
    - winzorize button --> DONE
    - remove y filter --> DONE
    - winzorize button and hist option should go to "under the hood"