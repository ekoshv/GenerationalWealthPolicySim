import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns # For distribution plots
import pygad
import threading
import time
import platform
from datetime import datetime
import io
import xlsxwriter
import json # For save/load config

matplotlib.use("TkAgg")
sns.set_theme(style="whitegrid") # Optional: Use seaborn styling for plots

# --- Tooltip Helper Class (Unchanged) ---
class ToolTip:
    # ... (Tooltip class code remains the same as corrected previously) ...
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.enter, add='+')
        self.widget.bind("<Leave>", self.leave, add='+')
        self.widget.bind("<ButtonPress>", self.leave, add='+') # Hide on click

    def enter(self, event=None):
        if not self.widget.winfo_exists(): return
        try:
            # Simple position relative to widget
            x = self.widget.winfo_rootx() + 10
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        except: # Fallback
             x = self.widget.winfo_rootx() + 50
             y = self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        # Position check
        screen_width = self.widget.winfo_screenwidth(); screen_height = self.widget.winfo_screenheight()
        est_width = 210; est_height = 60 # Approximate size
        if x + est_width > screen_width: x = screen_width - est_width - 5
        if y + est_height > screen_height: y = self.widget.winfo_rooty() - est_height - 5
        if x < 0: x = 5
        if y < 0: y = 5
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="#FFFFE0", relief="solid", borderwidth=1, justify='left', wraplength=200, font=("TkDefaultFont", 8))
        label.pack(ipadx=1)

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# --- Economic Logic Constants and Functions ---
LIFESPAN = 85
GENERATIONS = 3
MONTE_CARLO_RUNS = 2000 # Default runs for final results

# Default Household Params (can be skill-adjusted)
DEFAULT_INCOME_MEAN = 35000
DEFAULT_INCOME_STD = 8000
DEFAULT_RETIREMENT_INCOME = 12000
DEFAULT_YOUTH_CONSUMPTION = 8000
DEFAULT_RETIREMENT_CONSUMPTION = 18000
DEFAULT_WAGE_GROWTH = 0.01
DEFAULT_UNEMPLOYMENT_PROB = 0.05

# --- Skill Level Adjustments ---
# Simple multiplicative factors (example, adjust as needed)
SKILL_FACTORS = {
    "Low": {"income_mean": 0.7, "income_std": 0.9, "wage_growth": 0.8, "unemployment_prob": 1.5},
    "Medium": {"income_mean": 1.0, "income_std": 1.0, "wage_growth": 1.0, "unemployment_prob": 1.0},
    "High": {"income_mean": 1.5, "income_std": 1.2, "wage_growth": 1.2, "unemployment_prob": 0.5},
}


# Tax Brackets, Wealth Tax, Inheritance Tax (Unchanged)
INCOME_TAX_BRACKETS = [(30000, 0.20), (60000, 0.30), (999999999, 0.40)]
WEALTH_TAX_THRESHOLD = 300000; WEALTH_TAX_RATE = 0.02
def progressive_inheritance_tax(inherited):
    if inherited < 300000: return 0.10
    elif inherited < 700000: return 0.20
    else: return 0.35

# Default Fiscal Params
DEFAULT_CAPITAL_GAINS_TAX_RATE = 0.15
DEFAULT_CONSUMPTION_TAX_RATE = 0.10
DEFAULT_PAYROLL_TAX_RATE = 0.07
DEFAULT_BASIC_EXEMPTION = 10000.0
DEFAULT_FISCAL_BONUS_RATE = 0.0001
DEFAULT_PROPERTY_TAX_RATE = 0.01

# bracket_income_tax, apply_asset_allocation (Modified for Dynamic Allocation)
def bracket_income_tax(income, basic_exemption=DEFAULT_BASIC_EXEMPTION):
    # ... (implementation unchanged) ...
    taxable_income = max(0, income - basic_exemption)
    remaining = taxable_income; total_tax = 0; lower_bound = 0
    for upper_bound, rate in INCOME_TAX_BRACKETS:
        taxable = min(remaining, upper_bound - lower_bound)
        if taxable > 0: total_tax += taxable * rate; remaining -= taxable
        lower_bound = upper_bound
        if remaining <= 0: break
    return total_tax

def get_current_stocks_fraction(age, enable_dynamic_allocation, initial_fraction, final_fraction, start_de_risk_age):
    """Calculates target stocks fraction based on age and dynamic settings."""
    if not enable_dynamic_allocation or age < start_de_risk_age:
        return initial_fraction
    # Linear de-risking from start_de_risk_age to LIFESPAN
    retirement_span = LIFESPAN - start_de_risk_age
    if retirement_span <= 0: return final_fraction # Avoid division by zero if start age is >= lifespan
    progress = min(1.0, (age - start_de_risk_age) / retirement_span)
    return initial_fraction + (final_fraction - initial_fraction) * progress


def apply_asset_allocation(asset, age, base_interest, inflation, shock_prob, shock_impact,
                           enable_dynamic_allocation, initial_stocks_fraction, final_stocks_fraction, start_de_risk_age):
    """ Calculates real return considering dynamic asset allocation and shocks. """
    # Determine current stocks fraction based on age
    stocks_fraction = get_current_stocks_fraction(age, enable_dynamic_allocation, initial_stocks_fraction, final_stocks_fraction, start_de_risk_age)

    stock_nominal = base_interest + 0.03 # Simplified stock premium
    bond_nominal = base_interest
    nominal_return_rate = stocks_fraction * stock_nominal + (1 - stocks_fraction) * bond_nominal
    real_return_rate = nominal_return_rate - inflation

    # Apply shock if occurs
    if np.random.rand() < shock_prob:
        real_return_rate *= (1 - shock_impact)

    return real_return_rate # Return the rate


# simulate_three_generations (Modified for Skill Levels, Dynamic Allocation, Explicit Costs)
def simulate_three_generations(
    progress_callback=None, stop_event=None, n_runs=MONTE_CARLO_RUNS, **kwargs
):
    """ Simulate three generations with enhanced features. """
    # --- Extract Parameters ---
    # Economic
    inflation = kwargs.get('inflation', 0.05)
    base_interest = kwargs.get('base_interest', 0.02)
    interest_trend = kwargs.get('interest_trend', 0.0)
    shock_prob = kwargs.get('shock_prob', 0.15)
    shock_impact = kwargs.get('shock_impact', 0.30)
    productivity_growth = kwargs.get('productivity_growth', 0.00)

    # Household / Behavior / Demographics
    skill_level = kwargs.get('skill_level', "Medium") # NEW
    n_children = kwargs.get('n_children', 2)
    forced_saving = kwargs.get('forced_saving', 0.10)
    overspending_prob = kwargs.get('overspending_prob', 0.05)
    overspending_factor = kwargs.get('overspending_factor', 1.2)
    enable_housing = kwargs.get('housing_choice', False)
    mortgage_payment = kwargs.get('mortgage_payment', 5000)

    # Dynamic Allocation Params (NEW)
    enable_dynamic_allocation = kwargs.get('enable_dynamic_allocation', False)
    initial_stocks_fraction = kwargs.get('initial_stocks_fraction', 0.5)
    final_stocks_fraction = kwargs.get('final_stocks_fraction', 0.1)
    start_de_risk_age = kwargs.get('start_de_risk_age', 55)

    # Policy Levers
    guarantee = kwargs.get('guarantee', 10000)
    ubc = kwargs.get('ubc', 20000)
    unemployment_benefit = kwargs.get('unemployment_benefit', 5000)
    old_age_medical = kwargs.get('old_age_medical', 2000)

    # Fiscal Policy
    basic_exemption = kwargs.get('basic_exemption', DEFAULT_BASIC_EXEMPTION)
    capital_gains_tax_rate = kwargs.get('capital_gains_tax_rate', DEFAULT_CAPITAL_GAINS_TAX_RATE)
    consumption_tax_rate = kwargs.get('consumption_tax_rate', DEFAULT_CONSUMPTION_TAX_RATE)
    payroll_tax_rate = kwargs.get('payroll_tax_rate', DEFAULT_PAYROLL_TAX_RATE)
    property_tax_rate = kwargs.get('property_tax_rate', DEFAULT_PROPERTY_TAX_RATE)
    fiscal_bonus_rate = kwargs.get('fiscal_bonus_rate', DEFAULT_FISCAL_BONUS_RATE)

    # Skill Adjustments (NEW)
    skill_adj = SKILL_FACTORS.get(skill_level, SKILL_FACTORS["Medium"])
    adj_income_mean = DEFAULT_INCOME_MEAN * skill_adj["income_mean"]
    adj_income_std = DEFAULT_INCOME_STD * skill_adj["income_std"]
    adj_wage_growth = DEFAULT_WAGE_GROWTH * skill_adj["wage_growth"]
    adj_unemployment_prob = DEFAULT_UNEMPLOYMENT_PROB * skill_adj["unemployment_prob"]
    # Ensure prob is valid
    adj_unemployment_prob = np.clip(adj_unemployment_prob, 0.0, 1.0)

    # --- Simulation Setup ---
    generation_wealth = np.zeros((n_runs, GENERATIONS))
    total_gov_cost_direct = 0.0 # UBC, Guarantee, Unemp Benefits
    total_gov_revenue = 0.0

    # --- Simulation Loop ---
    for run in range(n_runs):
        if stop_event and stop_event.is_set(): return np.zeros((n_runs, GENERATIONS)), 0.0, 0.0 # Stop early
        assets_next_gen = 0
        for gen in range(GENERATIONS):
            age_assets = assets_next_gen
            yearly_income_mean = adj_income_mean # Use skill-adjusted mean
            gen_gov_rev = 0.0
            gen_gov_cost_direct = 0.0 # Direct costs for this generation

            for age in range(LIFESPAN):
                income_tax, payroll_tax, cons_tax, prop_tax, wealth_tax, cap_gains_tax = 0, 0, 0, 0, 0, 0
                unemp_benefit_paid_this_year = 0

                if age < 22: # Youth
                    income = 0
                    consumption = DEFAULT_YOUTH_CONSUMPTION
                    net_income = 0 # No income tax on zero income
                elif age < 65: # Working years
                    yearly_income_mean *= (1 + adj_wage_growth) # Adjusted wage growth
                    yearly_income_mean *= (1 + productivity_growth * (age / LIFESPAN))
                    raw_income = max(0, np.random.normal(yearly_income_mean, adj_income_std)) # Adjusted std dev

                    is_unemployed = np.random.rand() < adj_unemployment_prob # Adjusted prob
                    if is_unemployed:
                        income = unemployment_benefit
                        payroll_tax = 0
                        unemp_benefit_paid_this_year = unemployment_benefit # Track cost
                    else:
                        income = raw_income
                        payroll_tax = income * payroll_tax_rate

                    income_tax = bracket_income_tax(income, basic_exemption)
                    net_income = income - income_tax - payroll_tax

                    # Consumption logic
                    base_rate = 0.55 if income < 60000 else 0.60 # Based on actual income this year
                    c_rate = max(0, base_rate - forced_saving)
                    consumption = income * c_rate if income > 0 else DEFAULT_YOUTH_CONSUMPTION # Base consumption
                    consumption = max(0, consumption)
                    if np.random.rand() < overspending_prob: consumption *= overspending_factor
                    if enable_housing:
                        consumption += mortgage_payment
                        prop_tax = mortgage_payment * property_tax_rate
                else: # Retirement
                    income = DEFAULT_RETIREMENT_INCOME
                    consumption = DEFAULT_RETIREMENT_CONSUMPTION + old_age_medical
                    income_tax = bracket_income_tax(income, basic_exemption)
                    net_income = income - income_tax
                    if enable_housing:
                        consumption += mortgage_payment
                        prop_tax = mortgage_payment * property_tax_rate

                # Apply Consumption Tax
                cons_tax = consumption * consumption_tax_rate
                effective_consumption = consumption + cons_tax
                savings = net_income - effective_consumption

                # Policy interventions affecting assets/costs
                ubc_paid_this_year = 0
                if age == 21 and ubc > 0:
                    age_assets += ubc
                    ubc_paid_this_year = ubc # Track cost

                # Apply Wealth Tax
                if age_assets > WEALTH_TAX_THRESHOLD:
                    wealth_tax = (age_assets - WEALTH_TAX_THRESHOLD) * WEALTH_TAX_RATE
                    age_assets -= wealth_tax

                # Asset Growth (using new dynamic allocation)
                curr_interest = base_interest + age * interest_trend
                real_return_rate = apply_asset_allocation(
                    age_assets, age, curr_interest, inflation, shock_prob, shock_impact,
                    enable_dynamic_allocation, initial_stocks_fraction, final_stocks_fraction, start_de_risk_age
                )

                return_base = age_assets + savings
                gain_loss = return_base * real_return_rate
                new_assets = return_base + gain_loss

                # Apply Capital Gains Tax
                if gain_loss > 0:
                    cap_gains_tax = gain_loss * capital_gains_tax_rate
                    new_assets -= cap_gains_tax

                age_assets = max(0, new_assets) # Floor assets at 0

                # Accumulate taxes and direct costs for the generation/run
                gen_gov_rev += (income_tax + payroll_tax + cons_tax + prop_tax + wealth_tax + cap_gains_tax)
                gen_gov_cost_direct += (ubc_paid_this_year + unemp_benefit_paid_this_year) # ADD Unemp cost


            # End-of-life inheritance & guarantee
            n_children_safe = max(1, int(n_children))
            inherited = age_assets / n_children_safe
            i_tax_rate = progressive_inheritance_tax(inherited)
            inheritance_tax = inherited * i_tax_rate
            inherited_after_tax = inherited - inheritance_tax

            guarantee_cost_this_gen = 0
            if inherited_after_tax < guarantee:
                shortfall = guarantee - inherited_after_tax
                guarantee_cost_this_gen = shortfall
                inherited_after_tax = guarantee

            gen_gov_rev += inheritance_tax
            gen_gov_cost_direct += guarantee_cost_this_gen

            # Fiscal feedback
            gov_balance_gen = gen_gov_rev - gen_gov_cost_direct
            final_inheritance_per_child = max(0, inherited_after_tax + gov_balance_gen * fiscal_bonus_rate)

            # Store generation end wealth and accumulate totals
            generation_wealth[run, gen] = age_assets
            total_gov_revenue += gen_gov_rev
            total_gov_cost_direct += gen_gov_cost_direct

            if gen < GENERATIONS - 1:
                assets_next_gen = final_inheritance_per_child

        # --- Progress Update ---
        if progress_callback and (run + 1) % (n_runs // 20 or 1) == 0:
            progress_callback((run + 1) / n_runs * 100)

    if progress_callback: progress_callback(100)

    # Return full wealth matrix for distribution analysis
    generation_wealth = np.nan_to_num(generation_wealth)
    total_gov_cost_direct = np.nan_to_num(total_gov_cost_direct)
    total_gov_revenue = np.nan_to_num(total_gov_revenue)

    # Return total direct costs and total revenue
    return generation_wealth, total_gov_cost_direct, total_gov_revenue


# simulate_single_lifecycle (Modified similarly)
def simulate_single_lifecycle(**kwargs):
    """ Simulate single lifecycle with enhanced features. """
    # --- Extract Parameters ---
    inflation = kwargs.get('inflation', 0.05)
    base_interest = kwargs.get('base_interest', 0.02)
    interest_trend = kwargs.get('interest_trend', 0.0)
    shock_prob = kwargs.get('shock_prob', 0.15)
    shock_impact = kwargs.get('shock_impact', 0.30)
    productivity_growth = kwargs.get('productivity_growth', 0.00)
    skill_level = kwargs.get('skill_level', "Medium")
    n_children = kwargs.get('n_children', 2) # Less relevant but keep for consistency
    forced_saving = kwargs.get('forced_saving', 0.10)
    overspending_prob = kwargs.get('overspending_prob', 0.05)
    overspending_factor = kwargs.get('overspending_factor', 1.2)
    enable_housing = kwargs.get('housing_choice', False)
    mortgage_payment = kwargs.get('mortgage_payment', 5000)
    enable_dynamic_allocation = kwargs.get('enable_dynamic_allocation', False)
    initial_stocks_fraction = kwargs.get('initial_stocks_fraction', 0.5)
    final_stocks_fraction = kwargs.get('final_stocks_fraction', 0.1)
    start_de_risk_age = kwargs.get('start_de_risk_age', 55)
    guarantee = kwargs.get('guarantee', 10000) # Not really used here
    ubc = kwargs.get('ubc', 20000)
    unemployment_benefit = kwargs.get('unemployment_benefit', 5000)
    old_age_medical = kwargs.get('old_age_medical', 2000)
    basic_exemption = kwargs.get('basic_exemption', DEFAULT_BASIC_EXEMPTION)
    capital_gains_tax_rate = kwargs.get('capital_gains_tax_rate', DEFAULT_CAPITAL_GAINS_TAX_RATE)
    consumption_tax_rate = kwargs.get('consumption_tax_rate', DEFAULT_CONSUMPTION_TAX_RATE)
    payroll_tax_rate = kwargs.get('payroll_tax_rate', DEFAULT_PAYROLL_TAX_RATE)
    property_tax_rate = kwargs.get('property_tax_rate', DEFAULT_PROPERTY_TAX_RATE) # Used for gov calc, not individual here

    # Skill Adjustments
    skill_adj = SKILL_FACTORS.get(skill_level, SKILL_FACTORS["Medium"])
    adj_income_mean = DEFAULT_INCOME_MEAN * skill_adj["income_mean"]
    adj_income_std = DEFAULT_INCOME_STD * skill_adj["income_std"]
    adj_wage_growth = DEFAULT_WAGE_GROWTH * skill_adj["wage_growth"]
    adj_unemployment_prob = np.clip(DEFAULT_UNEMPLOYMENT_PROB * skill_adj["unemployment_prob"], 0.0, 1.0)


    # --- Simulation Setup ---
    ages = np.arange(LIFESPAN)
    income_array = np.zeros(LIFESPAN); consumption_array = np.zeros(LIFESPAN)
    savings_array = np.zeros(LIFESPAN); assets_array = np.zeros(LIFESPAN)
    cumulative_savings_array = np.zeros(LIFESPAN)
    age_assets = 0
    yearly_income_mean = adj_income_mean

    # --- Simulation Loop ---
    for i, age in enumerate(ages):
        income_tax, payroll_tax, wealth_tax, cap_gains_tax = 0, 0, 0, 0

        if age < 22: # Youth
            income = 0; consumption = DEFAULT_YOUTH_CONSUMPTION; net_income = 0
        elif age < 65: # Working years
            yearly_income_mean *= (1 + adj_wage_growth)
            yearly_income_mean *= (1 + productivity_growth * (age / LIFESPAN))
            raw_income = max(0, np.random.normal(yearly_income_mean, adj_income_std))
            is_unemployed = np.random.rand() < adj_unemployment_prob
            if is_unemployed:
                income = unemployment_benefit; payroll_tax = 0
            else:
                income = raw_income; payroll_tax = income * payroll_tax_rate
            income_tax = bracket_income_tax(income, basic_exemption)
            net_income = income - income_tax - payroll_tax

            base_rate = 0.55 if income < 60000 else 0.60
            c_rate = max(0, base_rate - forced_saving)
            consumption = income * c_rate if income > 0 else DEFAULT_YOUTH_CONSUMPTION
            consumption = max(0, consumption)
            if np.random.rand() < overspending_prob: consumption *= overspending_factor
            if enable_housing: consumption += mortgage_payment
        else: # Retirement
            income = DEFAULT_RETIREMENT_INCOME
            consumption = DEFAULT_RETIREMENT_CONSUMPTION + old_age_medical
            income_tax = bracket_income_tax(income, basic_exemption)
            net_income = income - income_tax
            if enable_housing: consumption += mortgage_payment

        # Consumption Tax effect on savings
        cons_tax = consumption * consumption_tax_rate
        effective_consumption = consumption + cons_tax
        savings = net_income - effective_consumption

        # Policy interventions
        if age == 21 and ubc > 0: age_assets += ubc
        if age_assets > WEALTH_TAX_THRESHOLD:
            wealth_tax = (age_assets - WEALTH_TAX_THRESHOLD) * WEALTH_TAX_RATE
            age_assets -= wealth_tax

        # Asset Growth
        curr_interest = base_interest + i * interest_trend # Use 'i' for index-based trend
        real_return_rate = apply_asset_allocation(
            age_assets, age, curr_interest, inflation, shock_prob, shock_impact,
            enable_dynamic_allocation, initial_stocks_fraction, final_stocks_fraction, start_de_risk_age
        )
        return_base = age_assets + savings
        gain_loss = return_base * real_return_rate
        new_assets = return_base + gain_loss
        if gain_loss > 0:
            cap_gains_tax = gain_loss * capital_gains_tax_rate
            new_assets -= cap_gains_tax
        age_assets = max(0, new_assets)

        # Store results
        income_array[i] = income; consumption_array[i] = consumption # Base consumption
        savings_array[i] = savings; assets_array[i] = age_assets
        cumulative_savings_array[i] = (cumulative_savings_array[i - 1] if i > 0 else 0) + savings

    # NaN check
    arrays = [income_array, consumption_array, savings_array, assets_array, cumulative_savings_array]
    for arr in arrays: np.nan_to_num(arr, copy=False)

    return ages, income_array, consumption_array, savings_array, assets_array, cumulative_savings_array


##############################################################################
#                          Modern Tkinter UI/UX
##############################################################################

class ModernPolicyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Enhanced Economic Policy Simulator")
        self.geometry("1100x850") # Wider for more controls/results

        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # --- Parameter Maps ---
        self._setup_parameter_maps()

        # --- Main Structure ---
        main_app_frame = ttk.Frame(self)
        main_app_frame.pack(fill=tk.BOTH, expand=True)

        # Paned Window for Resizable Panels
        self.paned_window = ttk.PanedWindow(main_app_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        # --- Left Frame (Scrollable Controls) ---
        self._setup_scrollable_controls_frame()

        # --- Right Frame (Results Notebook) ---
        self._setup_results_frame()

        # Add frames to PanedWindow
        # Initial sash position can be set if needed, weights provide starting ratio
        self.paned_window.add(self.controls_outer_frame, weight=1)
        self.paned_window.add(self.results_outer_frame, weight=2) # Give results more space initially

        # --- Controls Content ---
        self.param_vars = {}
        self.opt_ranges = {}
        self._create_controls_content() # Populate the scrollable frame

        # --- Status Bar ---
        self._create_status_bar(main_app_frame)

        # --- Simulation Thread Handling ---
        self.simulation_thread = None
        self.stop_event = threading.Event()
        self.current_results_data = None # Store latest results for reporting

        # Defer plot creation slightly
        self.after(100, self._create_plots)

    def _setup_parameter_maps(self):
        """ Defines mappings between internal names and UI labels. """
        self.internal_to_ui_map = {
            "inflation": "Inflation", "shock_prob": "ShockProbability", "shock_impact": "ShockImpact",
            "forced_saving": "ForcedSaving", "guarantee": "Guarantee", "ubc": "UBC",
            "n_children": "ChildrenPerFamily", "unemployment_prob": "DEFAULT_UnemploymentProb", # Base value if not skill adjusted
            "unemployment_benefit": "UnemploymentBenefit", "base_interest": "BaseInterest", "interest_trend": "InterestTrend",
            "productivity_growth": "ProductivityGrowth", "old_age_medical": "OldAgeMedical",
            "housing_choice": "EnableHousing", "mortgage_payment": "MortgagePayment",
            "capital_gains_tax_rate": "CapitalGainsTax", "consumption_tax_rate": "ConsumptionTax",
            "payroll_tax_rate": "PayrollTax", "basic_exemption": "BasicExemption",
            "fiscal_bonus_rate": "FiscalBonus", "property_tax_rate": "PropertyTax",
            # New/Modified
            "skill_level": "SkillLevel",
            "enable_dynamic_allocation": "EnableDynamicAllocation",
            "initial_stocks_fraction": "InitialStocksFraction",
            "final_stocks_fraction": "FinalStocksFraction",
            "start_de_risk_age": "StartDeRiskAge",
        }
        self.ui_to_internal_map = {v: k for k, v in self.internal_to_ui_map.items()}
        # Add optimization settings params manually if needed by get_params
        self.ui_to_internal_map.update({
            "N_Samples_Random": "n_samples_random", "GA_Generations": "ga_generations",
            "GA_Population": "ga_population", "GA_ParentsMating": "ga_parents_mating",
            "N_Runs_In_GA": "n_runs_in_ga",
        })

    def _setup_scrollable_controls_frame(self):
        """ Creates the resizable, scrollable left frame using a Canvas. """
        # Min size prevents it from collapsing too much
        self.controls_outer_frame = ttk.Frame(self.paned_window, width=380, height=600, style='Controls.TFrame')
        self.controls_outer_frame.pack_propagate(False) # Needed if using paned window? Maybe not. Let's try without.
        # self.controls_outer_frame.pack_propagate(True)


        self.controls_canvas = tk.Canvas(self.controls_outer_frame, borderwidth=0, highlightthickness=0) # No background needed if frame has it
        controls_scrollbar = ttk.Scrollbar(self.controls_outer_frame, orient="vertical", command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls_scrollbar.pack(side="right", fill="y")
        self.controls_canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame = ttk.Frame(self.controls_canvas, padding=5, style='Scrollable.TFrame') # Content frame
        self.scrollable_frame_window_id = self.controls_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.controls_canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mouse_wheel(self.controls_canvas)
        self._bind_mouse_wheel(self.scrollable_frame)
        # Also bind to children recursively? Or rely on bind_all? bind_all should cover it.

    def _setup_results_frame(self):
        """ Creates the right frame holding results (summary, plots, log). """
        self.results_outer_frame = ttk.Frame(self.paned_window, padding=5)

        # Top: Summary Frame
        summary_frame = ttk.LabelFrame(self.results_outer_frame, text="Summary Results", padding=10)
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        self.summary_labels = {}
        metrics = ["Avg Wealth Gen 1", "Avg Wealth Gen 2", "Avg Wealth Gen 3",
                   "Avg Gov Direct Cost", "Avg Gov Revenue", "Avg Gov Balance", "Cumulative Gov Balance", # Renamed Cost, added Cumulative
                   "Scenario Info", "Last Report"]
        for i, metric in enumerate(metrics):
            summary_frame.grid_columnconfigure(1, weight=1)
            ttk.Label(summary_frame, text=f"{metric}:").grid(row=i, column=0, sticky="w", padx=5, pady=1)
            wraplen = 180 if metric != "Last Report" else 350
            val_label = ttk.Label(summary_frame, text="N/A", width=25, anchor="w", justify=tk.LEFT, wraplength=wraplen)
            val_label.grid(row=i, column=1, sticky="ew", padx=5, pady=1)
            self.summary_labels[metric] = val_label

        # Middle: Results Notebook (Plots, Log)
        self._create_results_notebook(self.results_outer_frame)


    def _create_results_notebook(self, parent):
         """ Creates the notebook for plots and logs. """
         results_notebook = ttk.Notebook(parent)
         results_notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

         # --- Plot Tabs ---
         self.plot_notebook = ttk.Notebook(results_notebook) # Nested notebook for plots

         self.plot_frame_3gen = ttk.Frame(self.plot_notebook)
         self.plot_frame_lifecycle = ttk.Frame(self.plot_notebook)
         self.plot_frame_distribution = ttk.Frame(self.plot_notebook) # NEW Distribution Tab

         self.plot_notebook.add(self.plot_frame_3gen, text=' 3-Gen Wealth ')
         self.plot_notebook.add(self.plot_frame_lifecycle, text=' Single Lifecycle ')
         self.plot_notebook.add(self.plot_frame_distribution, text=' Wealth Distribution ')

         results_notebook.add(self.plot_notebook, text=' Plots ') # Add plot notebook to main results notebook

         # --- Log Tab ---
         log_frame = ttk.Frame(results_notebook, padding=5)
         results_notebook.add(log_frame, text=' Detailed Log ')

         self.results_box = tk.Text(log_frame, width=70, height=10, wrap=tk.WORD, borderwidth=0, font=("TkDefaultFont", 8))
         log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.results_box.yview)
         self.results_box['yscrollcommand'] = log_scroll.set
         log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
         self.results_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _create_controls_content(self):
        """ Populates the scrollable controls frame. """
        # --- Top Row: Presets & Save/Load ---
        config_frame = ttk.Frame(self.scrollable_frame)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(config_frame, text="Config:").pack(side=tk.LEFT, padx=(0, 5))
        # Presets
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(config_frame, textvariable=self.preset_var,
                                    values=["Baseline", "High Inflation", "Generous Welfare", "High Inequality Start"],
                                    state="readonly", width=18)
        preset_combo.pack(side=tk.LEFT, padx=5)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)
        ToolTip(preset_combo, "Load predefined parameter sets.")
        self.preset_var.set("Baseline") # Default selection

        # Save/Load Buttons
        load_btn = ttk.Button(config_frame, text="Load", width=6, command=self.load_config)
        load_btn.pack(side=tk.LEFT, padx=(10, 2))
        ToolTip(load_btn, "Load parameters from a JSON file.")
        save_btn = ttk.Button(config_frame, text="Save", width=6, command=self.save_config)
        save_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(save_btn, "Save current parameters to a JSON file.")


        # --- Main Controls Notebook ---
        controls_notebook = ttk.Notebook(self.scrollable_frame)
        controls_notebook.pack(fill=tk.X, expand=True, pady=(5,0))

        # Tab 1: Simulation Setup
        setup_tab = ttk.Frame(controls_notebook, padding=5)
        controls_notebook.add(setup_tab, text=' Simulation Setup ')
        self._create_setup_tab(setup_tab)

        # Tab 2: Optimization Settings
        opt_tab = ttk.Frame(controls_notebook, padding=5)
        controls_notebook.add(opt_tab, text=' Optimization ')
        self._create_optimization_tab(opt_tab)

        # Tab 3: Sensitivity Analysis (Placeholder)
        sens_tab = ttk.Frame(controls_notebook, padding=5)
        controls_notebook.add(sens_tab, text=' Sensitivity ')
        self._create_sensitivity_tab(sens_tab)


        # --- Action Buttons Frame ---
        action_frame = ttk.Frame(self.scrollable_frame, padding=(0, 5))
        action_frame.pack(fill=tk.X, pady=(10, 0))
        # Buttons (use grid for equal spacing maybe?)
        action_frame.columnconfigure((0, 1, 2), weight=1)

        run_sim_btn = ttk.Button(action_frame, text="Run Simulation", command=self.run_simulation_threaded)
        run_sim_btn.grid(row=0, column=0, padx=2, pady=5, sticky='ew')
        ToolTip(run_sim_btn, "Run standard simulation and generate report.")

        run_worst_btn = ttk.Button(action_frame, text="Worst Econ Search", command=self.run_worst_economy_threaded)
        run_worst_btn.grid(row=0, column=1, padx=2, pady=5, sticky='ew')
        ToolTip(run_worst_btn, "Search for worst economic conditions allowing positive wealth,\nupdate UI with best params, and generate report.")

        run_ga_btn = ttk.Button(action_frame, text="Run GA Optimization", command=self.run_ga_threaded)
        run_ga_btn.grid(row=0, column=2, padx=2, pady=5, sticky='ew')
        ToolTip(run_ga_btn, "Use GA to optimize parameters, update UI with best params,\nand generate report.")

    def _create_status_bar(self, parent):
        """ Creates the status bar at the bottom. """
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN, padding=2)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_bar = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=150, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)


    def _create_plots(self):
        """ Create plot figures and canvases. """
        try:
            figsize = (5.5, 3.8) # Common figure size
            # 3-Gen Plot
            self.fig_3gen, self.ax_3gen = plt.subplots(figsize=figsize)
            self.canvas_3gen = FigureCanvasTkAgg(self.fig_3gen, master=self.plot_frame_3gen)
            self.canvas_widget_3gen = self.canvas_3gen.get_tk_widget()
            self.toolbar_3gen = NavigationToolbar2Tk(self.canvas_3gen, self.plot_frame_3gen, pack_toolbar=False)
            self.toolbar_3gen.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas_widget_3gen.pack(fill=tk.BOTH, expand=True)
            self.toolbar_3gen.update()

            # Lifecycle Plot
            self.fig_lifecycle, self.ax_lifecycle = plt.subplots(figsize=figsize)
            self.canvas_lifecycle = FigureCanvasTkAgg(self.fig_lifecycle, master=self.plot_frame_lifecycle)
            self.canvas_widget_lifecycle = self.canvas_lifecycle.get_tk_widget()
            self.toolbar_lifecycle = NavigationToolbar2Tk(self.canvas_lifecycle, self.plot_frame_lifecycle, pack_toolbar=False)
            self.toolbar_lifecycle.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas_widget_lifecycle.pack(fill=tk.BOTH, expand=True)
            self.toolbar_lifecycle.update()

            # Distribution Plot
            self.fig_dist, self.ax_dist = plt.subplots(figsize=figsize)
            self.canvas_dist = FigureCanvasTkAgg(self.fig_dist, master=self.plot_frame_distribution)
            self.canvas_widget_dist = self.canvas_dist.get_tk_widget()
            self.toolbar_dist = NavigationToolbar2Tk(self.canvas_dist, self.plot_frame_distribution, pack_toolbar=False)
            self.toolbar_dist.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas_widget_dist.pack(fill=tk.BOTH, expand=True)
            self.toolbar_dist.update()

            self.clear_results() # Draw initial empty plots
        except Exception as e:
            self.log_message(f"Error creating plots: {e}")
            messagebox.showerror("Plot Error", f"Failed to initialize plots: {e}")


    # --- Scrolling Helpers (Unchanged) ---
    def _on_frame_configure(self, event=None):
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))
    def _on_canvas_configure(self, event):
        self.controls_canvas.itemconfig(self.scrollable_frame_window_id, width=event.width)
    def _on_mouse_wheel(self, event):
        # Add check if scrolling is actually needed
        bbox = self.controls_canvas.bbox("all")
        scroll_needed = bbox and bbox[3] > self.controls_canvas.winfo_height()
        # Only scroll if needed OR if scrolling up when already at top (allows "bouncing")
        current_view = self.controls_canvas.yview()
        scrolling_up = (platform.system() == 'Windows' and event.delta > 0) or \
                       (platform.system() == 'Darwin' and event.delta > 0) or \
                       (platform.system() == 'Linux' and event.num == 4)

        if scroll_needed or (scrolling_up and current_view[0] > 0.0):
             if platform.system() == 'Windows': delta = -int(event.delta / 120)
             elif platform.system() == 'Darwin': delta = -event.delta
             else: delta = -1 if event.num == 4 else 1 if event.num == 5 else 0
             self.controls_canvas.yview_scroll(delta, "units")

    def _bind_mouse_wheel(self, widget):
        # Bind recursively to ensure all children are covered
        widget.bind_all("<MouseWheel>", self._on_mouse_wheel, add='+') # Windows/macOS
        widget.bind_all("<Button-4>", self._on_mouse_wheel, add='+')   # Linux scroll up
        widget.bind_all("<Button-5>", self._on_mouse_wheel, add='+')   # Linux scroll down
        # for child in widget.winfo_children():
        #     self._bind_mouse_wheel(child) # Might cause too many bindings? bind_all preferred.

    # --- Parameter Input & UI Creation ---
    def _add_param(self, parent, label, default, tooltip, row, var_type=tk.DoubleVar, width=10, use_grid=True, col=1, **kwargs):
        """ Helper to add labeled param. Uses grid layout by default. """
        l = ttk.Label(parent, text=label + ":")
        if use_grid: l.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        else: l.pack(side=tk.LEFT, padx=5, pady=2)
        ToolTip(l, tooltip)

        # Ensure variable exists only once
        if label not in self.param_vars:
            var = var_type()
            var.set(default)
            self.param_vars[label] = var
        else:
            var = self.param_vars[label] # Reuse existing var if called again

        if var_type == tk.BooleanVar:
            widget = ttk.Checkbutton(parent, variable=var, **kwargs)
        elif var_type == tk.StringVar and 'values' in kwargs: # Use Combobox for StringVar with values
            widget = ttk.Combobox(parent, textvariable=var, values=kwargs.pop('values'), state='readonly', width=width-2, **kwargs)
        else:
            widget = ttk.Entry(parent, textvariable=var, width=width, **kwargs)

        if use_grid: widget.grid(row=row, column=col, sticky="ew", padx=5, pady=2)
        else: widget.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)

        ToolTip(widget, tooltip)
        # self._bind_mouse_wheel(widget) # Covered by bind_all on parent/canvas
        # self._bind_mouse_wheel(l)
        return row + 1

    def _create_setup_tab(self, tab):
        """ Creates the content for the Simulation Setup tab using grid. """
        # Use grid layout within the frames for better alignment
        col_weight = 1 # Make input column expandable

        # --- Economic Conditions ---
        econ_frame = ttk.LabelFrame(tab, text="Economic Conditions", padding=10)
        econ_frame.pack(fill="x", expand=True, padx=5, pady=5)
        econ_frame.columnconfigure(1, weight=col_weight)
        r = 0
        r = self._add_param(econ_frame, "Inflation", 0.05, "Annual inflation rate (e.g., 0.02 for 2%)", r)
        r = self._add_param(econ_frame, "BaseInterest", 0.02, "Base annual nominal interest rate for bonds", r)
        r = self._add_param(econ_frame, "InterestTrend", 0.0, "Linear trend added to base interest per year of age", r)
        r = self._add_param(econ_frame, "ShockProbability", 0.15, "Probability of a negative economic shock each year", r)
        r = self._add_param(econ_frame, "ShockImpact", 0.30, "Severity of shock (multiplicative reduction in returns, e.g., 0.3 for 30%)", r)
        r = self._add_param(econ_frame, "ProductivityGrowth", 0.00, "Economy-wide productivity growth (affects wages)", r)

        # --- Household Behavior / Demographics ---
        behav_frame = ttk.LabelFrame(tab, text="Household Behavior / Demographics", padding=10)
        behav_frame.pack(fill="x", expand=True, padx=5, pady=5)
        behav_frame.columnconfigure(1, weight=col_weight)
        r = 0
        r = self._add_param(behav_frame, "SkillLevel", "Medium", "Agent skill level (affects income, unemployment)", r,
                           var_type=tk.StringVar, values=list(SKILL_FACTORS.keys()), width=12) # NEW Skill Level
        r = self._add_param(behav_frame, "DEFAULT_UnemploymentProb", DEFAULT_UNEMPLOYMENT_PROB, "Base annual unemployment probability (adjusted by skill)", r) # Clarified label
        r = self._add_param(behav_frame, "DEFAULT_WageGrowth", DEFAULT_WAGE_GROWTH, "Base annual wage growth (adjusted by skill)", r) # Clarified label
        r = self._add_param(behav_frame, "ChildrenPerFamily", 2, "Number of children inheriting (integer)", r, var_type=tk.IntVar)
        r = self._add_param(behav_frame, "ForcedSaving", 0.10, "Minimum saving rate enforced (reduces consumption rate)", r)
        r = self._add_param(behav_frame, "OverspendingProb", 0.05, "Probability of consuming more than planned in a working year", r)
        r = self._add_param(behav_frame, "OverspendingFactor", 1.2, "Multiplier for consumption when overspending occurs (e.g., 1.2 = 20% more)", r)

        # --- Asset Allocation ---
        asset_frame = ttk.LabelFrame(tab, text="Asset Allocation", padding=10)
        asset_frame.pack(fill="x", expand=True, padx=5, pady=5)
        asset_frame.columnconfigure(1, weight=col_weight)
        r = 0
        # Dynamic Allocation replaces single StocksFraction
        r = self._add_param(asset_frame, "EnableDynamicAllocation", False, "Enable age-based de-risking?", r, var_type=tk.BooleanVar)
        r = self._add_param(asset_frame, "InitialStocksFraction", 0.6, "Stocks fraction before de-risking starts", r)
        r = self._add_param(asset_frame, "FinalStocksFraction", 0.1, "Stocks fraction target at end of life", r)
        r = self._add_param(asset_frame, "StartDeRiskAge", 55, "Age at which de-risking begins", r, var_type=tk.IntVar)


        # --- Housing ---
        housing_frame = ttk.LabelFrame(tab, text="Housing", padding=10)
        housing_frame.pack(fill="x", expand=True, padx=5, pady=5)
        housing_frame.columnconfigure(1, weight=col_weight)
        r = 0
        r = self._add_param(housing_frame, "EnableHousing", False, "Include fixed mortgage payments in consumption?", r, var_type=tk.BooleanVar)
        r = self._add_param(housing_frame, "MortgagePayment", 5000, "Annual mortgage payment (if EnableHousing is True)", r)


        # --- Policy Levers ---
        policy_frame = ttk.LabelFrame(tab, text="Policy Levers", padding=10)
        policy_frame.pack(fill="x", expand=True, padx=5, pady=5)
        policy_frame.columnconfigure(1, weight=col_weight)
        r = 0
        r = self._add_param(policy_frame, "Guarantee", 10000, "Minimum inheritance amount guaranteed by the government", r)
        r = self._add_param(policy_frame, "UBC", 20000, "Universal Basic Capital amount given at age 21", r)
        r = self._add_param(policy_frame, "UnemploymentBenefit", 5000, "Annual benefit received when unemployed", r)
        r = self._add_param(policy_frame, "OldAgeMedical", 2000, "Additional annual consumption cost during retirement for medical needs", r)

        # --- Fiscal Policy (Taxes & Transfers) ---
        fiscal_frame = ttk.LabelFrame(tab, text="Fiscal Policy (Taxes & Transfers)", padding=10)
        fiscal_frame.pack(fill="x", expand=True, padx=5, pady=5)
        fiscal_frame.columnconfigure(1, weight=col_weight)
        r = 0
        r = self._add_param(fiscal_frame, "BasicExemption", 10000, "Income amount exempt from income tax", r)
        r = self._add_param(fiscal_frame, "CapitalGainsTax", 0.15, "Tax rate on positive investment returns", r)
        r = self._add_param(fiscal_frame, "ConsumptionTax", 0.10, "Tax rate on consumption spending (VAT)", r)
        r = self._add_param(fiscal_frame, "PayrollTax", 0.07, "Tax rate on wage income (Social Security/Payroll)", r)
        r = self._add_param(fiscal_frame, "PropertyTax", 0.01, "Tax rate on mortgage payment (if housing enabled)", r)
        r = self._add_param(fiscal_frame, "FiscalBonus", 0.0001, "Rate at which generational government surplus/deficit affects inheritance", r)
        # Fixed tax info (as before)
        info_row = r
        ttk.Separator(fiscal_frame, orient=tk.HORIZONTAL).grid(row=info_row, column=0, columnspan=2, sticky='ew', pady=5)
        info_row += 1
        ttk.Label(fiscal_frame, text="Fixed Taxes/Thresholds:", font='-weight bold').grid(row=info_row, column=0, columnspan=2, sticky="w", padx=5, pady=(2,2))
        info_row += 1
        ttk.Label(fiscal_frame, text="  Income Brackets:", foreground="grey").grid(row=info_row, column=0, sticky="w", padx=5, pady=0)
        ttk.Label(fiscal_frame, text=f"<{INCOME_TAX_BRACKETS[0][0]}:{INCOME_TAX_BRACKETS[0][1]*100:.0f}%, <{INCOME_TAX_BRACKETS[1][0]}:{INCOME_TAX_BRACKETS[1][1]*100:.0f}%, >:{INCOME_TAX_BRACKETS[2][1]*100:.0f}%", foreground="grey").grid(row=info_row, column=1, sticky="w", padx=5, pady=0)
        info_row += 1
        ttk.Label(fiscal_frame, text="  Wealth Tax:", foreground="grey").grid(row=info_row, column=0, sticky="w", padx=5, pady=0)
        ttk.Label(fiscal_frame, text=f"{WEALTH_TAX_RATE*100:.0f}% above £{WEALTH_TAX_THRESHOLD:,.0f}", foreground="grey").grid(row=info_row, column=1, sticky="w", padx=5, pady=0)
        info_row += 1
        ttk.Label(fiscal_frame, text="  Inheritance Tax:", foreground="grey").grid(row=info_row, column=0, sticky="w", padx=5, pady=0)
        ttk.Label(fiscal_frame, text="10%/20%/35% progressive", foreground="grey").grid(row=info_row, column=1, sticky="w", padx=5, pady=0)

    def _add_opt_range(self, parent, param_label, default_min, default_max, tooltip, row):
        # ... (Code for _add_opt_range is the same) ...
        l = ttk.Label(parent, text=param_label + ":")
        l.grid(row=row, column=0, sticky="w", padx=5, pady=3)
        ToolTip(l, tooltip)

        f = ttk.Frame(parent)
        f.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        parent.columnconfigure(1, weight=1)

        min_var = tk.DoubleVar(value=default_min); max_var = tk.DoubleVar(value=default_max)

        ttk.Label(f, text="Min:").pack(side=tk.LEFT, padx=(0, 2))
        min_entry = ttk.Entry(f, textvariable=min_var, width=8)
        min_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(f, text="Max:").pack(side=tk.LEFT, padx=(0, 2))
        max_entry = ttk.Entry(f, textvariable=max_var, width=8)
        max_entry.pack(side=tk.LEFT)

        if param_label not in self.opt_ranges:
             self.opt_ranges[param_label] = {"min": min_var, "max": max_var}

        ToolTip(min_entry, f"Min value for {param_label} during optimization."); ToolTip(max_entry, f"Max value for {param_label} during optimization.")
        return row + 1

    def _create_optimization_tab(self, tab):
        # ... (Code for _create_optimization_tab is the same) ...
        gen_opt_frame = ttk.LabelFrame(tab, text="General Optimization Settings", padding=10)
        gen_opt_frame.pack(fill="x", expand=True, padx=5, pady=5)
        gen_opt_frame.columnconfigure(1, weight=1)
        r = 0
        r = self._add_param(gen_opt_frame, "N_Samples_Random", 50, "Number of random samples for 'Worst Economy Search'", r, var_type=tk.IntVar)
        r = self._add_param(gen_opt_frame, "GA_Generations", 50, "Number of generations for Genetic Algorithm", r, var_type=tk.IntVar)
        r = self._add_param(gen_opt_frame, "GA_Population", 20, "Population size per generation in GA", r, var_type=tk.IntVar)
        r = self._add_param(gen_opt_frame, "GA_ParentsMating", 10, "Number of solutions selected as parents in GA", r, var_type=tk.IntVar)
        r = self._add_param(gen_opt_frame, "N_Runs_In_GA", 200, "Monte Carlo runs per fitness evaluation inside GA (lower for speed)", r, var_type=tk.IntVar)

        range_frame = ttk.LabelFrame(tab, text="Parameter Ranges for Optimization", padding=10)
        range_frame.pack(fill="x", expand=True, padx=5, pady=5)
        r = 0
        # Add ranges for dynamically allocated parameters too? Only if optimized. Sticking to original for now.
        r = self._add_opt_range(range_frame, "Inflation", 0.0, 0.1, "Range for Inflation rate", r)
        r = self._add_opt_range(range_frame, "ShockProbability", 0.0, 0.5, "Range for Shock Probability", r)
        r = self._add_opt_range(range_frame, "ShockImpact", 0.0, 0.5, "Range for Shock Impact", r)
        r = self._add_opt_range(range_frame, "Guarantee", 5000.0, 15000.0, "Range for Inheritance Guarantee (used in GA only)", r)
        r = self._add_opt_range(range_frame, "UBC", 0.0, 30000.0, "Range for Universal Basic Capital (used in GA only)", r)

        note_label = ttk.Label(tab, text="Note: Ranges apply ONLY to 'Worst Economy Search' & 'GA Optimization'.", justify=tk.LEFT, foreground="grey", wraplength=300)
        note_label.pack(fill="x", padx=10, pady=(10,5))


    def _create_sensitivity_tab(self, tab):
        """ Creates the placeholder tab for sensitivity analysis. """
        ttk.Label(tab, text="Sensitivity Analysis (Under Development)", font="-weight bold").pack(pady=10)
        ttk.Label(tab, text="This section will allow varying one or two parameters \n"
                            "and plotting their impact on selected outcomes.", justify=tk.CENTER).pack(pady=5)

        param1_frame = ttk.Frame(tab, padding=5)
        param1_frame.pack(fill=tk.X, pady=2)
        ttk.Label(param1_frame, text="Parameter 1:").pack(side=tk.LEFT, padx=5)
        # TODO: Populate combobox with parameter names
        ttk.Combobox(param1_frame, values=["Inflation", "UBC", "..."], state='readonly', width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(param1_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(param1_frame, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(param1_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(param1_frame, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(param1_frame, text="Steps:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(param1_frame, width=5).pack(side=tk.LEFT, padx=5)

        param2_frame = ttk.Frame(tab, padding=5)
        param2_frame.pack(fill=tk.X, pady=2)
        # ... (Similar widgets for Parameter 2, optional) ...

        output_frame = ttk.Frame(tab, padding=5)
        output_frame.pack(fill=tk.X, pady=2)
        ttk.Label(output_frame, text="Output Metric:").pack(side=tk.LEFT, padx=5)
        # TODO: Populate combobox with output metrics
        ttk.Combobox(output_frame, values=["Avg Wealth Gen 3", "Gov Balance", "..."], state='readonly', width=25).pack(side=tk.LEFT, padx=5)

        run_sens_btn = ttk.Button(tab, text="Run Sensitivity Analysis", state=tk.DISABLED) # Disabled for now
        run_sens_btn.pack(pady=20)
        ToolTip(run_sens_btn, "Run simulations across the specified parameter ranges and plot results (feature under development).")


    # --- Config & Presets ---
    def _get_preset_params(self, preset_name):
         """ Returns a dictionary of parameters for a given preset name. """
         # Define baseline using current defaults or specified values
         baseline = {
             "inflation": 0.05, "shock_prob": 0.15, "shock_impact": 0.30, "forced_saving": 0.10,
             "guarantee": 10000, "ubc": 20000, "n_children": 2, "unemployment_benefit": 5000,
             "base_interest": 0.02, "interest_trend": 0.0, "productivity_growth": 0.00,
             "old_age_medical": 2000, "housing_choice": False, "mortgage_payment": 5000,
             "basic_exemption": 10000, "capital_gains_tax_rate": 0.15, "consumption_tax_rate": 0.10,
             "payroll_tax_rate": 0.07, "property_tax_rate": 0.01, "fiscal_bonus_rate": 0.0001,
             "skill_level": "Medium", "enable_dynamic_allocation": False, "initial_stocks_fraction": 0.6,
             "final_stocks_fraction": 0.1, "start_de_risk_age": 55,
             "DEFAULT_UnemploymentProb": DEFAULT_UNEMPLOYMENT_PROB, "DEFAULT_WageGrowth": DEFAULT_WAGE_GROWTH,
             "overspending_prob": 0.05, "overspending_factor": 1.2
         }
         if preset_name == "Baseline":
             return baseline
         elif preset_name == "High Inflation":
             p = baseline.copy(); p.update({"inflation": 0.10, "base_interest": 0.04}); return p
         elif preset_name == "Generous Welfare":
              p = baseline.copy(); p.update({"guarantee": 15000, "ubc": 30000, "unemployment_benefit": 7000, "basic_exemption": 12000}); return p
         elif preset_name == "High Inequality Start": # Example using skill level
               p = baseline.copy(); p.update({"skill_level": "Low"}); return p # Could also add variations in starting assets etc.
         else:
             return baseline # Default

    def _on_preset_selected(self, event=None):
        """ Loads parameters when a preset is selected from the combobox. """
        preset_name = self.preset_var.get()
        if not preset_name: return
        params_to_load = self._get_preset_params(preset_name)
        self.update_ui_params(params_to_load)
        self.log_message(f"Loaded preset: {preset_name}")

    def save_config(self):
        """ Saves current parameters to a JSON file. """
        params = self.get_params()
        if params is None: return # Error handled in get_params

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Configuration As"
        )
        if not filepath: return # User cancelled

        try:
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=4)
            self.log_message(f"Configuration saved to: {filepath}")
        except Exception as e:
            self.log_message(f"Error saving configuration: {e}")
            messagebox.showerror("Save Error", f"Failed to save configuration file:\n{e}")

    def load_config(self):
         """ Loads parameters from a JSON file. """
         filepath = filedialog.askopenfilename(
             filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
             title="Load Configuration"
         )
         if not filepath: return

         try:
             with open(filepath, 'r') as f:
                 loaded_params = json.load(f)
             # Validate loaded params? Basic check: is it a dict?
             if not isinstance(loaded_params, dict):
                 raise ValueError("Invalid configuration file format.")

             self.update_ui_params(loaded_params)
             self.log_message(f"Configuration loaded from: {filepath}")
             self.preset_var.set("") # Clear preset selection after loading custom file

         except Exception as e:
             self.log_message(f"Error loading configuration: {e}")
             messagebox.showerror("Load Error", f"Failed to load configuration file:\n{e}")

    def get_opt_bounds(self, param_label):
        """Get min/max bounds for a specific parameter from optimization settings."""
        # Ensure opt_ranges exists (it should be initialized in __init__)
        if not hasattr(self, 'opt_ranges'):
             messagebox.showerror("Internal Error", "opt_ranges dictionary not initialized.")
             self.status_var.set("Error: Internal opt_ranges missing")
             return None, None

        range_vars = self.opt_ranges.get(param_label)
        if not range_vars:
            # Check if maybe it was defined in param_vars (if it's also a base param)
            base_var = self.param_vars.get(param_label)
            if base_var:
                 messagebox.showerror("Config Error", f"Optimization range for '{param_label}' not defined in Optimization Settings tab.")
                 self.status_var.set(f"Error: Opt range missing for {param_label}")
                 return None, None
            else:
                 messagebox.showerror("Config Error", f"Parameter '{param_label}' not found in UI for optimization range.")
                 self.status_var.set(f"Error: Missing opt range for {param_label}")
                 return None, None

        try:
            lo = range_vars["min"].get()
            hi = range_vars["max"].get()
            if lo >= hi:
                 raise ValueError(f"Min value ({lo}) must be less than Max value ({hi}) for {param_label}.")
            return lo, hi
        except (tk.TclError, ValueError) as e:
             messagebox.showerror("Input Error", f"Invalid optimization range for {param_label}: {e}")
             self.status_var.set(f"Error: Invalid opt range for {param_label}")
             return None, None # Indicate error

    # --- Parameter Parsing & Update ---
    def get_params(self):
        """ Parse all parameters from UI, including new ones. """
        p = {}
        for label, var in self.param_vars.items():
            try:
                internal_name = self.ui_to_internal_map.get(label)
                if internal_name:
                    p[internal_name] = var.get()
            except (tk.TclError, ValueError) as e:
                 messagebox.showerror("Input Error", f"Invalid value for parameter: {label}\n{e}")
                 self.status_var.set(f"Error: Invalid value for {label}")
                 return None
        # Manually convert types for specific internal params if needed
        if 'n_children' in p: p['n_children'] = int(p.get('n_children', 2))
        if 'start_de_risk_age' in p: p['start_de_risk_age'] = int(p.get('start_de_risk_age', 55))
        # ... ensure other IntVars are handled if not using tk.IntVar directly ...

        # Add non-UI fixed params if they are ever needed elsewhere (unlikely now)
        # p['lifespan'] = LIFESPAN
        return p

    def update_ui_params(self, params_to_set):
        """ Updates UI fields based on a parameter dictionary. """
        # Use self.after to ensure updates happen in the main thread if called from bg thread
        self.after(0, self._update_ui_params_sync, params_to_set)

    def _update_ui_params_sync(self, params_to_set):
        """ Synchronous part of updating UI params. """
        self.log_message("Updating UI parameters...")
        updated_count = 0
        for internal_name, value in params_to_set.items():
            ui_label = self.internal_to_ui_map.get(internal_name)
            if ui_label and ui_label in self.param_vars:
                try:
                    var = self.param_vars[ui_label]
                    current_val = None
                    try: current_val = var.get() # Check current value before setting
                    except: pass

                    # Only set if the value is actually different (prevents unnecessary trace calls)
                    # Need type comparison too
                    set_needed = True
                    if current_val is not None:
                        try: # Compare values appropriately
                            if isinstance(var, tk.DoubleVar) and abs(float(current_val) - float(value)) < 1e-9: set_needed=False
                            elif isinstance(var, tk.IntVar) and int(current_val) == int(value): set_needed=False
                            elif isinstance(var, tk.BooleanVar) and bool(current_val) == bool(value): set_needed=False
                            elif isinstance(var, tk.StringVar) and str(current_val) == str(value): set_needed=False
                        except: pass # If comparison fails, assume set is needed

                    if set_needed:
                         if isinstance(var, tk.DoubleVar): var.set(float(value))
                         elif isinstance(var, tk.IntVar): var.set(int(value))
                         elif isinstance(var, tk.BooleanVar): var.set(bool(value))
                         else: var.set(str(value)) # Handles StringVar/Combobox
                         updated_count += 1
                except Exception as e:
                    print(f"Warning: Could not update UI for {ui_label}: {e}") # Log to console might be less intrusive

        if updated_count > 0: self.log_message(f"Updated {updated_count} UI parameters.")
        # else: self.log_message("No UI parameters needed updating.")


    # --- Results Display ---
    def clear_results(self):
        """ Clears previous results from the UI. """
        self.results_box.delete("1.0", tk.END)
        for key, label in self.summary_labels.items(): label.config(text="N/A")
        self.current_results_data = None # Clear stored results

        # Clear plots safely, checking if axes exist
        plot_axes = [getattr(self, ax_name, None) for ax_name in ['ax_3gen', 'ax_lifecycle', 'ax_dist']]
        plot_canvases = [getattr(self, canvas_name, None) for canvas_name in ['canvas_3gen', 'canvas_lifecycle', 'canvas_dist']]
        titles = ["3-Generation Wealth", "Single Lifecycle", "Wealth Distribution (Last Gen)"]

        for i, ax in enumerate(plot_axes):
             if ax:
                 ax.clear()
                 ax.set_title(titles[i], fontsize=10)
                 ax.grid(True, linestyle='--', alpha=0.6)
                 ax.figure.tight_layout(pad=0.5)
                 if plot_canvases[i]: plot_canvases[i].draw_idle()

    def update_summary(self, results):
        """ Updates the summary labels with results, including new metrics. """
        self.current_results_data = results # Store for potential later use (e.g., delayed report)

        wealth = results.get("avg_wealth", [0, 0, 0]); wealth = (list(wealth) + [0, 0, 0])[:3]
        cost = results.get("avg_gov_cost", 0) # Now direct costs
        rev = results.get("avg_gov_revenue", 0)
        balance = results.get("avg_gov_balance", 0)
        cum_balance = results.get("cumulative_gov_balance", 0) # NEW
        info = results.get("info", ""); report_path = results.get("report_file", "N/A")

        self.summary_labels["Avg Wealth Gen 1"].config(text=f"£{wealth[0]:,.2f}")
        self.summary_labels["Avg Wealth Gen 2"].config(text=f"£{wealth[1]:,.2f}")
        self.summary_labels["Avg Wealth Gen 3"].config(text=f"£{wealth[2]:,.2f}")
        self.summary_labels["Avg Gov Direct Cost"].config(text=f"£{cost:,.2f}") # Updated label
        self.summary_labels["Avg Gov Revenue"].config(text=f"£{rev:,.2f}")
        self.summary_labels["Avg Gov Balance"].config(text=f"£{balance:,.2f}")
        self.summary_labels["Cumulative Gov Balance"].config(text=f"£{cum_balance:,.2f}") # NEW
        self.summary_labels["Scenario Info"].config(text=info)
        self.summary_labels["Last Report"].config(text=report_path)

    def plot_results(self, results):
        """ Updates all embedded plots, including distribution. """
        # Ensure plots are created
        if not all(hasattr(self, name) for name in ['ax_3gen', 'ax_lifecycle', 'ax_dist']):
            self.log_message("Warning: Plots not initialized, skipping plot update.")
            return

        self.current_results_data = results # Store latest results

        # --- 3-Gen Plot ---
        # ... (Plotting logic unchanged) ...
        avg_wealth = results.get("avg_wealth")
        self.ax_3gen.clear()
        if avg_wealth is not None and len(avg_wealth) > 0:
            generations = range(1, len(avg_wealth) + 1)
            self.ax_3gen.plot(generations, avg_wealth, marker='o', color='steelblue', linestyle='-', linewidth=1.5, markersize=5)
            self.ax_3gen.set_xticks(generations)
        self.ax_3gen.set_title(results.get("plot_title_3gen", "Avg Final Wealth (3 Gens)"), fontsize=10)
        self.ax_3gen.set_xlabel("Generation", fontsize=9); self.ax_3gen.set_ylabel("Wealth (£)", fontsize=9)
        self.ax_3gen.tick_params(axis='both', which='major', labelsize=8)
        self.ax_3gen.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
        self.ax_3gen.grid(True, linestyle='--', alpha=0.6); self.fig_3gen.tight_layout(pad=0.5)
        if hasattr(self, 'canvas_3gen'): self.canvas_3gen.draw_idle()


        # --- Lifecycle Plot ---
        # ... (Plotting logic unchanged) ...
        single_lc_data = results.get("single_lifecycle")
        self.ax_lifecycle.clear()
        if single_lc_data:
            ages, inc, cons, sav, ast, cum_sav = single_lc_data
            self.ax_lifecycle.plot(ages, ast, label='Assets', color='darkorange', linewidth=1.5)
            self.ax_lifecycle.plot(ages, cum_sav, label='Cum. Savings', color='firebrick', linestyle=':', linewidth=1)
            self.ax_lifecycle.plot(ages, inc, label='Income', color='mediumseagreen', linestyle='--', linewidth=1)
            self.ax_lifecycle.plot(ages, cons, label='Consumption', color='lightcoral', linestyle='-.', linewidth=1)
            params_used = results.get("params", {})
            if params_used.get("ubc", 0) > 0: self.ax_lifecycle.axvline(x=21, color='green', linestyle='--', linewidth=0.8, label='UBC@21')
            self.ax_lifecycle.axvline(x=65, color='dimgrey', linestyle='--', linewidth=0.8, label='Retire@65')
            self.ax_lifecycle.legend(fontsize='x-small', loc='best')
        self.ax_lifecycle.set_title(results.get("plot_title_lc", "Single Lifecycle Simulation"), fontsize=10)
        self.ax_lifecycle.set_xlabel("Age", fontsize=9); self.ax_lifecycle.set_ylabel("Value (£)", fontsize=9)
        self.ax_lifecycle.tick_params(axis='both', which='major', labelsize=8)
        self.ax_lifecycle.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
        self.ax_lifecycle.grid(True, linestyle='--', alpha=0.6)
        if self.ax_lifecycle.has_data(): self.ax_lifecycle.set_ylim(bottom=min(0, self.ax_lifecycle.get_ylim()[0] ))
        self.fig_lifecycle.tight_layout(pad=0.5)
        if hasattr(self, 'canvas_lifecycle'): self.canvas_lifecycle.draw_idle()

        # --- Distribution Plot (NEW) ---
        self.ax_dist.clear()
        full_wealth_data = results.get("full_wealth_data") # Expecting the n_runs x GENERATIONS array
        if full_wealth_data is not None and full_wealth_data.size > 0:
             last_gen_wealth = full_wealth_data[:, -1] # Wealth of the last generation across all runs
             # Use seaborn for a nicer plot (histogram or violin)
             sns.histplot(last_gen_wealth, kde=True, ax=self.ax_dist, stat="density", linewidth=0.5, color="skyblue")
             # Or: sns.violinplot(y=last_gen_wealth, ax=self.ax_dist, inner="quartile", color="lightcoral")
             self.ax_dist.set_title(f"Wealth Distribution (Gen {GENERATIONS})", fontsize=10)
             self.ax_dist.set_xlabel("Final Wealth (£)", fontsize=9)
             self.ax_dist.set_ylabel("Density" if isinstance(self.ax_dist.collections, list) else "Density", fontsize=9) # Check plot type for label
             self.ax_dist.tick_params(axis='both', which='major', labelsize=8)
             self.ax_dist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
        else:
             self.ax_dist.text(0.5, 0.5, 'No distribution data available', horizontalalignment='center', verticalalignment='center', transform=self.ax_dist.transAxes)

        self.ax_dist.grid(True, linestyle='--', alpha=0.6)
        self.fig_dist.tight_layout(pad=0.5)
        if hasattr(self, 'canvas_dist'): self.canvas_dist.draw_idle()


    # --- Logging ---
    def log_message(self, message):
         if hasattr(self, 'results_box') and self.results_box.winfo_exists():
             self.after(0, lambda m=message: self._log_update(m))
    def _log_update(self, message):
         if hasattr(self, 'results_box') and self.results_box.winfo_exists():
             # Simple rate limiting or length check could be added here if logging becomes excessive
             self.results_box.insert(tk.END, message + "\n")
             self.results_box.see(tk.END)


    # --- Excel Report Generation (Modified) ---
    def generate_excel_report(self, results_data, filename):
        """ Generates an Excel report with parameters, summary, plots, and log. """
        try:
            params = results_data.get("params", {})
            # Include new summary metrics
            summary = {
                "Avg Wealth Gen 1": results_data.get("avg_wealth", [0]*3)[0],
                "Avg Wealth Gen 2": results_data.get("avg_wealth", [0]*3)[1],
                "Avg Wealth Gen 3": results_data.get("avg_wealth", [0]*3)[2],
                "Avg Gov Direct Cost": results_data.get("avg_gov_cost", 0),
                "Avg Gov Revenue": results_data.get("avg_gov_revenue", 0),
                "Avg Gov Balance": results_data.get("avg_gov_balance", 0),
                "Cumulative Gov Balance": results_data.get("cumulative_gov_balance", 0), # NEW
                "Scenario Info": results_data.get("info", "")
            }
            opt_details = results_data.get("optimization_details", {})
            # Include distribution stats?
            full_wealth = results_data.get("full_wealth_data")
            if full_wealth is not None and full_wealth.size > 0:
                 last_gen_wealth = full_wealth[:, -1]
                 summary["Wealth Median (Last Gen)"] = np.median(last_gen_wealth)
                 summary["Wealth Std Dev (Last Gen)"] = np.std(last_gen_wealth)
                 # Gini coefficient could be added here if needed (requires calculation)


            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                workbook = writer.book
                # --- Parameters Sheet ---
                # Sort params alphabetically for consistency?
                params_list = sorted(params.items())
                params_df = pd.DataFrame(params_list, columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
                param_sheet = writer.sheets['Parameters']; param_sheet.set_column('A:A', 25); param_sheet.set_column('B:B', 15)

                # --- Summary Sheet ---
                summary_list = sorted(summary.items())
                summary_df = pd.DataFrame(summary_list, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                summary_sheet = writer.sheets['Summary']
                summary_sheet.set_column('A:A', 30); summary_sheet.set_column('B:B', 20)
                currency_format = workbook.add_format({'num_format': '£#,##0.00'})
                float_format = workbook.add_format({'num_format': '0.00'})
                for row_num, metric in enumerate(summary_df['Metric'], 1):
                    value = summary_df.iloc[row_num-1]['Value']
                    if isinstance(value, (int, float, np.number)):
                         # Apply currency format based on metric name heuristic
                         fmt = currency_format if ('Wealth' in metric or 'Gov' in metric or 'Median' in metric) else float_format
                         summary_sheet.write_number(row_num, 1, value, fmt)
                    else:
                         summary_sheet.write_string(row_num, 1, str(value)) # Write strings as strings


                # --- Optimization Details Sheet ---
                if opt_details:
                    opt_list = sorted(opt_details.items())
                    opt_df = pd.DataFrame(opt_list, columns=['Detail', 'Value'])
                    opt_df.to_excel(writer, sheet_name='Optimization Details', index=False)
                    opt_sheet = writer.sheets['Optimization Details']; opt_sheet.set_column('A:A', 25); opt_sheet.set_column('B:B', 20)


                # --- Plot Sheets ---
                plot_figures = {
                    '3-Gen Plot': getattr(self, 'fig_3gen', None),
                    'Lifecycle Plot': getattr(self, 'fig_lifecycle', None),
                    'Distribution Plot': getattr(self, 'fig_dist', None)
                }
                for sheet_name, fig in plot_figures.items():
                    plot_sheet = workbook.add_worksheet(sheet_name)
                    if fig:
                         img_buffer = io.BytesIO()
                         try:
                             fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                             img_buffer.seek(0)
                             plot_sheet.insert_image('B2', filename, {'image_data': img_buffer, 'x_scale': 0.9, 'y_scale': 0.9}) # Scale slightly
                         except Exception as e:
                              plot_sheet.write('A1', f"Error saving plot: {e}")
                              self.log_message(f"Warning: Failed to save plot '{sheet_name}' to Excel: {e}")
                    else:
                         plot_sheet.write('A1', "Plot not available.")


                # --- Detailed Log Sheet ---
                if hasattr(self, 'results_box'):
                    log_text = self.results_box.get("1.0", tk.END)
                    log_sheet = workbook.add_worksheet('Detailed Log')
                    log_lines = log_text.strip().split('\n')
                    text_wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
                    log_sheet.set_column('A:A', 120, text_wrap_format) # Wider log column
                    max_len = 32000 # Excel cell character limit approximation
                    for i, line in enumerate(log_lines):
                         # Truncate long lines to avoid Excel errors
                         log_sheet.write(i, 0, line[:max_len])

            self.log_message(f"Report saved to: {filename}")
            self.after(0, lambda f=filename: self.summary_labels["Last Report"].config(text=f)) # Update summary via lambda
            return True

        except Exception as e:
            import traceback
            error_msg = f"Error generating Excel report '{filename}': {e}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.after(0, lambda em=str(e): messagebox.showerror("Report Error", f"Could not generate Excel report:\n{em}"))
            self.after(0, lambda: self.summary_labels["Last Report"].config(text="Error generating report"))
            return False


    # --- Threading and Simulation Launchers ---
    def _start_simulation_thread(self, target_func, args=()):
        # ... (Code unchanged, ensures UI updates are scheduled via self.after) ...
        if self.simulation_thread and self.simulation_thread.is_alive():
            messagebox.showwarning("Busy", "Another simulation is already running.")
            return
        self.stop_event.clear()
        self.progress_bar['mode'] = 'determinate'
        self.after(0, self.progress_bar.config, {'value': 0})
        self.after(0, self.status_var.set, "Running...")
        self.after(0, lambda: self.summary_labels["Last Report"].config(text="N/A"))

        self.simulation_thread = threading.Thread(target=target_func, args=args, daemon=True)
        self.simulation_thread.start()
        self.after(100, self._check_simulation_thread)

    def _check_simulation_thread(self):
        # ... (Code unchanged) ...
         if self.simulation_thread and self.simulation_thread.is_alive():
            self.after(100, self._check_simulation_thread)
         else:
            self.after(0, self._finalize_simulation_status)

    def _finalize_simulation_status(self):
        # ... (Code unchanged) ...
        current_status = self.status_var.get()
        final_status = "Ready" # Default
        if "Running" in current_status:
             # Check if stopped only if it was running
             if self.stop_event.is_set():
                 final_status = "Stopped"
             # Otherwise, status might already be set to "Finished..."
             elif "Finished" not in current_status:
                  final_status = "Ready" # Or should be Finished? Logic depends on where status is finally set
             else:
                 final_status = current_status # Keep "Finished..."
        elif "Stopping" in current_status:
             final_status = "Stopped"
        else: # Was already Ready, Error, Finished, Stopped etc.
             final_status = current_status

        # Only update if needed and not already Error/Finished/Stopped from the worker thread's final step
        if self.status_var.get() != final_status and final_status == "Ready":
             self.status_var.set(final_status)

        self.progress_bar['value'] = 0
        self.progress_bar['mode'] = 'determinate'

    def stop_simulation(self):
         # ... (Code unchanged) ...
          if self.simulation_thread and self.simulation_thread.is_alive():
             self.stop_event.set()
             self.after(0, self.status_var.set, "Stopping...")
             self.log_message(">>> Stop request sent...")

    def update_progress(self, value):
         # ... (Code unchanged) ...
          self.after(0, lambda v=value: self.progress_bar.config(value=v))

    def run_simulation_threaded(self): self._start_simulation_thread(self.run_simulation_logic)
    def run_worst_economy_threaded(self): self._start_simulation_thread(self.run_worst_economy_logic)
    def run_ga_threaded(self): self._start_simulation_thread(self.run_ga_logic)


    # --- Core Logic Wrappers (Modified for new features & reporting) ---

    def run_simulation_logic(self):
        """ Runs standard simulation, updates UI, generates report. """
        self.after(0, self.clear_results)
        params = self.get_params();
        if params is None: return

        run_type = "Standard Simulation"
        self.log_message(f"=== Running {run_type} ===")
        # Selective logging of parameters
        log_params = {k:v for k,v in params.items() if not k.startswith(('n_samples', 'ga_', 'n_runs'))}
        self.log_message(f"Parameters: {json.dumps(log_params, indent=2)}")

        try:
            # --- Simulation ---
            full_wealth_data, total_gov_cost, total_gov_revenue = simulate_three_generations(
                n_runs=MONTE_CARLO_RUNS, progress_callback=self.update_progress, stop_event=self.stop_event, **params
            )
            if self.stop_event.is_set():
                self.log_message("Simulation stopped."); self.after(0, self.status_var.set, "Stopped"); return

            # --- Process Results ---
            avg_wealth = full_wealth_data.mean(axis=0) if full_wealth_data.size > 0 else np.zeros(GENERATIONS)
            avg_gov_cost = total_gov_cost / MONTE_CARLO_RUNS if MONTE_CARLO_RUNS > 0 else 0
            avg_gov_rev = total_gov_revenue / MONTE_CARLO_RUNS if MONTE_CARLO_RUNS > 0 else 0
            avg_gov_balance = avg_gov_rev - avg_gov_cost
            # Conceptual cumulative balance (sum over the runs, not true debt dynamics)
            cumulative_gov_balance = total_gov_revenue - total_gov_cost

            self.log_message("\n--- Results ---")
            self.log_message(f"Avg Final Wealth: Gen1={avg_wealth[0]:,.2f}, Gen2={avg_wealth[1]:,.2f}, Gen3={avg_wealth[2]:,.2f}")
            self.log_message(f"Avg Gov Direct Cost: £{avg_gov_cost:,.2f}")
            self.log_message(f"Avg Gov Revenue: £{avg_gov_rev:,.2f}")
            self.log_message(f"Avg Gov Balance: £{avg_gov_balance:,.2f}")
            self.log_message(f"Total Cumulative Balance (across runs): £{cumulative_gov_balance:,.2f}")

            # Run single lifecycle for plot data
            lc_data = simulate_single_lifecycle(**params)

            # --- Prepare Data & Update UI ---
            results_data = {
                "params": params, "full_wealth_data": full_wealth_data, "avg_wealth": avg_wealth,
                "avg_gov_cost": avg_gov_cost, "avg_gov_revenue": avg_gov_rev, "avg_gov_balance": avg_gov_balance,
                "cumulative_gov_balance": cumulative_gov_balance, # Add cumulative balance
                "single_lifecycle": lc_data,
                "plot_title_3gen": f"Avg. Wealth ({run_type})", "plot_title_lc": f"Lifecycle ({run_type})",
                "info": f"{run_type} Run", "type": run_type
            }
            # Schedule UI updates to happen in the main thread
            self.after(0, lambda d=results_data.copy(): self.update_summary(d))
            self.after(0, lambda d=results_data.copy(): self.plot_results(d))
            self.log_message("\nSimulation Complete.")
            self.after(0, self.status_var.set, f"Finished: {run_type}")

            # --- Generate Report ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"Simulation_Report_{run_type.replace(' ','')}_{timestamp}.xlsx"
            results_data["report_file"] = report_filename
            self.generate_excel_report(results_data, report_filename) # Run in this thread (usually fast enough)

        except Exception as e:
            # ... (Error handling unchanged) ...
            import traceback
            error_msg = f"\nError during {run_type}: {e}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.after(0, lambda em=str(e): messagebox.showerror("Simulation Error", f"An error occurred: {em}"))
            self.after(0, self.status_var.set, "Error")


    def run_worst_economy_logic(self):
        """ Runs worst economy search, updates UI with best, generates report. """
        self.after(0, self.clear_results)
        base_params = self.get_params();
        if base_params is None: return

        run_type = "Worst Economy Search"
        n_samples = base_params.get("n_samples_random", 50)
        n_runs_per_sample = max(1, MONTE_CARLO_RUNS // 20) # Fewer runs for faster search

        self.log_message(f"=== Running {run_type} ===")
        # ... (logging parameters) ...
        self.log_message(f"Number of Samples: {n_samples} (using {n_runs_per_sample} runs per sample)")

        try:
            # --- Get Bounds ---
            infl_lo, infl_hi = self.get_opt_bounds("Inflation")
            sp_lo, sp_hi = self.get_opt_bounds("ShockProbability")
            si_lo, si_hi = self.get_opt_bounds("ShockImpact")
            if infl_lo is None or sp_lo is None or si_lo is None: return
            self.log_message(f"Search Ranges: Inf[{infl_lo:.3f}-{infl_hi:.3f}], SP[{sp_lo:.3f}-{sp_hi:.3f}], SI[{si_lo:.3f}-{si_hi:.3f}]")

            # --- Search Loop ---
            # ... (search logic unchanged) ...
            best_index = -1e15; best_cost = 1e15
            best_scenario_params = None; found_valid = False
            self.after(0, self.progress_bar.config, {'mode': 'determinate'})
            for i in range(n_samples):
                 if self.stop_event.is_set(): self.log_message("Search stopped."); self.after(0, self.status_var.set, "Stopped"); return
                 progress = (i + 1) / n_samples * 100
                 self.update_progress(progress)
                 self.after(0, self.status_var.set, f"Running {run_type}: Sample {i+1}/{n_samples}")
                 trial_params = dict(base_params)
                 trial_params["inflation"] = random.uniform(infl_lo, infl_hi); trial_params["shock_prob"] = random.uniform(sp_lo, sp_hi); trial_params["shock_impact"] = random.uniform(si_lo, si_hi)
                 gen_wealth, gov_cost, _ = simulate_three_generations(n_runs=n_runs_per_sample, stop_event=None, **trial_params)
                 avg_wealth = gen_wealth.mean(axis=0)
                 if np.any(avg_wealth <= 0): continue
                 found_valid = True
                 bad_index = trial_params["inflation"] + trial_params["shock_prob"] + trial_params["shock_impact"]
                 current_cost = gov_cost / n_runs_per_sample if n_runs_per_sample > 0 else 0
                 if bad_index > best_index or (abs(bad_index - best_index) < 1e-9 and current_cost < best_cost):
                     best_index = bad_index; best_cost = current_cost; best_scenario_params = trial_params
            self.after(0, self.progress_bar.config, {'value': 100})

            # --- Process Results ---
            if not found_valid or best_scenario_params is None:
                self.log_message("\nNo valid scenario found."); self.after(0, self.status_var.set, f"Finished: {run_type} (No valid scenario)"); return

            self.log_message("\n--- Best Scenario Found ---")
            self.log_message(f"Best Badness Index: {best_index:.4f}")
            self.log_message(f"Params: Inf={best_scenario_params['inflation']:.4f}, SP={best_scenario_params['shock_prob']:.4f}, SI={best_scenario_params['shock_impact']:.4f}")

            # --- Rerun with Full Runs ---
            self.log_message("\nRunning final simulation with best parameters...")
            self.after(0, self.status_var.set, f"Running final simulation ({run_type})...")
            self.after(0, self.progress_bar.config, {'mode': 'indeterminate', 'value': 0})
            full_wealth_data, total_gov_cost, total_gov_revenue = simulate_three_generations(
                 n_runs=MONTE_CARLO_RUNS, stop_event=self.stop_event, progress_callback=None, **best_scenario_params
            )
            if self.stop_event.is_set(): self.log_message("Final run stopped."); self.after(0, self.status_var.set, "Stopped"); return

            avg_wealth_final = full_wealth_data.mean(axis=0) if full_wealth_data.size > 0 else np.zeros(GENERATIONS)
            avg_cost_final = total_gov_cost / MONTE_CARLO_RUNS if MONTE_CARLO_RUNS > 0 else 0
            avg_rev_final = total_gov_revenue / MONTE_CARLO_RUNS if MONTE_CARLO_RUNS > 0 else 0
            balance_final = avg_rev_final - avg_cost_final
            cumulative_gov_balance = total_gov_revenue - total_gov_cost
            lc_data = simulate_single_lifecycle(**best_scenario_params)

            # --- Prepare Data & Update UI ---
            opt_details_report = {
                "Type": run_type, "Samples Run": n_samples, "Best Badness Index": best_index,
                "Optimized Inflation": best_scenario_params['inflation'], "Optimized Shock Prob": best_scenario_params['shock_prob'],
                "Optimized Shock Impact": best_scenario_params['shock_impact'],
            }
            results_data = {
                "params": best_scenario_params, "full_wealth_data": full_wealth_data, "avg_wealth": avg_wealth_final,
                "avg_gov_cost": avg_cost_final, "avg_gov_revenue": avg_rev_final, "avg_gov_balance": balance_final,
                "cumulative_gov_balance": cumulative_gov_balance,
                "single_lifecycle": lc_data,
                "plot_title_3gen": f"Avg. Wealth ({run_type}, Idx={best_index:.3f})",
                "plot_title_lc": f"Lifecycle ({run_type} Scenario)", "info": f"{run_type} (Idx={best_index:.3f})",
                "type": run_type, "optimization_details": opt_details_report
            }
            # Schedule UI updates
            self.after(0, lambda d=results_data.copy(): self.update_summary(d))
            self.after(0, lambda d=results_data.copy(): self.plot_results(d))
            self.after(0, lambda p=best_scenario_params.copy(): self.update_ui_params(p)) # UPDATE UI PARAMS
            self.log_message(f"\n{run_type} Complete.")
            self.after(0, self.status_var.set, f"Finished: {run_type}")

            # --- Generate Report ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"Optimization_Report_{run_type.replace(' ','')}_{timestamp}.xlsx"
            results_data["report_file"] = report_filename
            self.generate_excel_report(results_data, report_filename)

        except Exception as e:
             # ... (Error handling unchanged) ...
            import traceback
            error_msg = f"\nError during {run_type}: {e}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.after(0, lambda em=str(e): messagebox.showerror("Search Error", f"An error occurred: {em}"))
            self.after(0, self.status_var.set, "Error")


    def run_ga_logic(self):
        """ Runs GA optimization, updates UI with best, generates report. """
        self.after(0, self.clear_results)
        base_params = self.get_params();
        if base_params is None: return

        run_type = "Genetic Algorithm"
        ga_generations = base_params.get("ga_generations", 50)
        ga_population = base_params.get("ga_population", 20)
        ga_parents_mating = base_params.get("ga_parents_mating", 10)
        n_runs_in_ga = max(1, base_params.get("n_runs_in_ga", 200))

        self.log_message(f"=== Running {run_type} Optimization ===")
        # ... (logging parameters) ...

        try:
            # --- Get Bounds ---
            infl_lo, infl_hi = self.get_opt_bounds("Inflation")
            sp_lo, sp_hi = self.get_opt_bounds("ShockProbability")
            si_lo, si_hi = self.get_opt_bounds("ShockImpact")
            guarantee_lo, guarantee_hi = self.get_opt_bounds("Guarantee")
            ubc_lo, ubc_hi = self.get_opt_bounds("UBC")
            if None in [infl_lo, sp_lo, si_lo, guarantee_lo, ubc_lo]: return

            gene_space = [
                {"low": infl_lo, "high": infl_hi}, {"low": sp_lo, "high": sp_hi}, {"low": si_lo, "high": si_hi},
                {"low": guarantee_lo, "high": guarantee_hi}, {"low": ubc_lo, "high": ubc_hi}
            ]
            num_genes = len(gene_space)
            self.log_message(f"Optimizing: Inf[{infl_lo:.3f}-{infl_hi:.3f}], SP[{sp_lo:.3f}-{sp_hi:.3f}], SI[{si_lo:.3f}-{si_hi:.3f}], Guar[{guarantee_lo:,.0f}-{guarantee_hi:,.0f}], UBC[{ubc_lo:,.0f}-{ubc_hi:,.0f}]")

            # --- Fitness & Callback ---
            self.ga_base_params = base_params; self.ga_n_runs_in_ga = n_runs_in_ga
            # ... (fitness_func implementation unchanged) ...
            def fitness_func(ga_instance, solution, solution_idx):
                 trial_params = dict(self.ga_base_params)
                 trial_params.update({"inflation": solution[0], "shock_prob": solution[1], "shock_impact": solution[2], "guarantee": solution[3], "ubc": solution[4]})
                 if self.stop_event.is_set(): return -1e9
                 try:
                     gen_wealth, gov_cost, gov_rev = simulate_three_generations(n_runs=self.ga_n_runs_in_ga, stop_event=None, **trial_params)
                 except Exception as e: return -1e8
                 avg_wealth = gen_wealth.mean(axis=0);
                 if np.any(avg_wealth <= 0): return -1e6
                 avg_gov_cost = gov_cost / self.ga_n_runs_in_ga if self.ga_n_runs_in_ga > 0 else 0
                 avg_gov_rev = gov_rev / self.ga_n_runs_in_ga if self.ga_n_runs_in_ga > 0 else 0
                 government_balance = avg_gov_rev - avg_gov_cost
                 if government_balance < 0: return -1e5 + (government_balance * 10)
                 bad_index = solution[0] + solution[1] + solution[2]; min_wealth = avg_wealth.min()
                 cost_penalty_factor = 1e-4; min_wealth_bonus_factor = 1e-5; balance_bonus_factor = 5e-6
                 fitness = (bad_index - (avg_gov_cost * cost_penalty_factor) + (min_wealth * min_wealth_bonus_factor) + (government_balance * balance_bonus_factor))
                 return fitness

            # ... (on_generation_callback implementation unchanged) ...
            def on_generation_callback(ga_instance):
                 gen = ga_instance.generations_completed
                 best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
                 progress = (gen / ga_generations) * 100
                 self.after(0, lambda g=gen, bf=best_fitness, p=progress: self._update_ga_status(g, ga_generations, bf, p))
                 if self.stop_event.is_set(): return "stop"
                 time.sleep(0.01)

            # --- Run GA ---
            self.after(0, self.progress_bar.config, {'mode': 'determinate', 'value': 0})
            ga_instance = pygad.GA(
                num_generations=ga_generations, num_parents_mating=ga_parents_mating, fitness_func=fitness_func,
                sol_per_pop=ga_population, num_genes=num_genes, gene_space=gene_space, mutation_percent_genes=20,
                mutation_type="random", crossover_type="single_point", keep_elitism=1, on_generation=on_generation_callback,
                stop_criteria="saturate_10"
            )
            ga_instance.run()
            if self.stop_event.is_set(): self.log_message("GA stopped."); self.after(0, self.status_var.set, "Stopped"); return
            self.after(0, self.progress_bar.config, {'value': 100})

            # --- Process GA Results ---
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            self.log_message("\n--- GA Optimization Finished ---")
            self.log_message(f"Best Solution Fitness: {solution_fitness:.4f}")
            best_params_ga = dict(base_params)
            best_params_ga.update({
                 "inflation": solution[0], "shock_prob": solution[1], "shock_impact": solution[2],
                 "guarantee": solution[3], "ubc": solution[4]
            })
            self.log_message(f"Best Params: Inf={solution[0]:.4f}, SP={solution[1]:.4f}, SI={solution[2]:.4f}, Guar={solution[3]:,.2f}, UBC={solution[4]:,.2f}")


            # --- Rerun with Full Runs ---
            self.log_message("\nRunning final simulation with optimized parameters...")
            self.after(0, self.status_var.set, f"Running final simulation ({run_type})...")
            self.after(0, self.progress_bar.config, {'mode': 'indeterminate', 'value': 0})
            full_wealth_data, total_gov_cost, total_gov_revenue = simulate_three_generations(
                n_runs=MONTE_CARLO_RUNS, stop_event=self.stop_event, **best_params_ga
            )
            if self.stop_event.is_set(): self.log_message("Final run stopped."); self.after(0, self.status_var.set, "Stopped"); return

            avg_wealth_final = full_wealth_data.mean(axis=0) if full_wealth_data.size > 0 else np.zeros(GENERATIONS)
            avg_cost_final = total_gov_cost / MONTE_CARLO_RUNS if MONTE_CARLO_RUNS > 0 else 0
            avg_rev_final = total_gov_revenue / MONTE_CARLO_RUNS if MONTE_CARLO_RUNS > 0 else 0
            balance_final = avg_rev_final - avg_cost_final
            cumulative_gov_balance = total_gov_revenue - total_gov_cost
            bad_index_final = solution[0] + solution[1] + solution[2]
            lc_data = simulate_single_lifecycle(**best_params_ga)
            self.log_message("\n--- Final Results with GA Parameters ---")
            # ... (log results) ...

            # --- Prepare Data & Update UI ---
            opt_details_report = {
                "Type": run_type, "Generations Run": ga_instance.generations_completed, "Best Fitness": solution_fitness,
                "Optimized Inflation": solution[0], "Optimized Shock Prob": solution[1], "Optimized Shock Impact": solution[2],
                "Optimized Guarantee": solution[3], "Optimized UBC": solution[4],
            }
            results_data = {
                "params": best_params_ga, "full_wealth_data": full_wealth_data, "avg_wealth": avg_wealth_final,
                "avg_gov_cost": avg_cost_final, "avg_gov_revenue": avg_rev_final, "avg_gov_balance": balance_final,
                "cumulative_gov_balance": cumulative_gov_balance,
                "single_lifecycle": lc_data,
                "plot_title_3gen": f"Avg. Wealth ({run_type}, Fit={solution_fitness:.2f})",
                "plot_title_lc": f"Lifecycle ({run_type} Scenario)", "info": f"{run_type} (Fit={solution_fitness:.2f})",
                "type": run_type, "optimization_details": opt_details_report
            }
            # Schedule UI updates
            self.after(0, lambda d=results_data.copy(): self.update_summary(d))
            self.after(0, lambda d=results_data.copy(): self.plot_results(d))
            self.after(0, lambda p=best_params_ga.copy(): self.update_ui_params(p)) # UPDATE UI PARAMS
            self.log_message(f"\n{run_type} Optimization Complete.")
            self.after(0, self.status_var.set, f"Finished: {run_type} Optimization")

            # --- Generate Report ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"Optimization_Report_{run_type.replace(' ','')}_{timestamp}.xlsx"
            results_data["report_file"] = report_filename
            self.generate_excel_report(results_data, report_filename)

            # Save GA plot
            try:
                 ga_plot_filename = f"GA_Fitness_Plot_{timestamp}.png"
                 ga_instance.plot_fitness(title="GA Fitness Progression", save_dir=ga_plot_filename)
                 self.log_message(f"GA fitness plot saved to {ga_plot_filename}")
            except Exception as plot_err: self.log_message(f"Warning: Could not save GA plot: {plot_err}")

        except Exception as e:
            # ... (Error handling unchanged) ...
            import traceback
            error_msg = f"\nError during {run_type} optimization: {e}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.after(0, lambda em=str(e): messagebox.showerror("GA Error", f"An error occurred: {em}"))
            self.after(0, self.status_var.set, "Error")


    # --- Helper for GA status update ---
    def _update_ga_status(self, gen, total_gens, fitness, progress):
         if hasattr(self, 'status_var') and self.status_var.get() != "Stopping...":
              self.status_var.set(f"Running GA: Gen {gen}/{total_gens} - Best Fit: {fitness:.4f}")
         if hasattr(self, 'progress_bar'): self.progress_bar.config(value=progress)


def main():
    # Set high DPI awareness for Windows if applicable
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass # Fails on non-Windows or if SetProcessDpiAwareness is not available
    app = ModernPolicyApp()
    app.mainloop()

if __name__ == "__main__":
    main()