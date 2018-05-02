# Eric Collins, Final Project, May the markov gods have mercy on my soul, shine their light on my 100% homework score, and bless me with a Pass

# requires newest version of SupportLib MarkovModel to be loaded in content root and  Anaconda to be installed locally and chosen as the interpreter
# Slightly modified versions were used, but defaults will work, some image labels may be incorrect

# import required
import numpy as numpy
import scr.MarkovClasses as MarkovCls
from enum import Enum
import scr.SamplePathClasses as PathCls
import scr.StatisticalClasses as StatCls
import scr.RandomVariantGenerators as rndClasses
import scr.FormatFunctions as F
import scr.StatisticalClasses as Stat
import scr.EconEvalClasses as Econ
import scr.EconEvalClasses as EconCls
import scr.FigureSupport as Figs

# Parameter classes, modified from ParameterClasses.py

class HealthStats(Enum):
    """ health states of patients """
    WELL = 0
    CANCER = 1
    CANCERDEATH = 2
    MASTECTOMY = 3
    POSTMASTECTOMY = 4
    MASTECTOMYCANCER = 5
    OTHERDEATH = 6

class Therapies(Enum):
    """ standard vs. population screening """
    STANDARD = 0
    POPULATION = 1

class _Parameters:

    def __init__(self, therapy):


        # selected therapy
        self._therapy = therapy

        # simulation time step
        self._delta_t = DELTA_T

        # calculate the adjusted discount rate
        self._adjDiscountRate = DISCOUNT * DELTA_T

        # initial health state
        self._initialHealthState = HealthStats.WELL

        # annual treatment cost
        if self._therapy == Therapies.STANDARD:
            self._annualTreatmentCost = ((TEST_COST * POP_SIZE * 0.13)/SIM_LENGTH)
        else:
            self._annualTreatmentCost = ((TEST_COST * POP_SIZE)/SIM_LENGTH)

         # transition probability matrix of the selected therapy
        self._prob_matrix = []
        if therapy == Therapies.STANDARD:
            self._prob_matrix[:], p = MarkovCls.continuous_to_discrete(standard_transition, DELTA_T)
        elif therapy == Therapies.POPULATION:
            self._prob_matrix[:], p = MarkovCls.continuous_to_discrete(population_transition, DELTA_T)

        # annual state costs and utilities
        self._annualStateCosts = []
        self._annualStateUtilities = []

    def get_initial_health_state(self):
        return self._initialHealthState

    def get_delta_t(self):
        return self._delta_t

    def get_adj_discount_rate(self):
        return self._adjDiscountRate

    def get_transition_prob(self, state):
        return self._prob_matrix[state.value]

    def get_annual_state_cost(self, state):
        if state == HealthStats.CANCERDEATH or state == HealthStats.OTHERDEATH:
            return 0
        else:
            return self._annualStateCosts[state.value]

    def get_annual_state_utility(self, state):
        if state == HealthStats.CANCERDEATH or state == HealthStats.OTHERDEATH:
            return 0
        else:
            return self._annualStateUtilities[state.value]

    def get_annual_treatment_cost(self):
        return self._annualTreatmentCost

class ParametersFixed(_Parameters):
    def __init__(self, therapy):

        # initialize the base class
        _Parameters.__init__(self, therapy)

        # annual state costs and utilities
        self._annualStateCosts = ANNUAL_STATE_COST
        self._annualStateUtilities = ANNUAL_STATE_UTILITY

# Markov classes, modified from MarkovModelClasses.py
class Patient:
    def __init__(self, id, parameters):
        """ initiates a patient
        :param id: ID of the patient
        :param parameters: parameter object
        """

        self._id = id
        # random number generator for this patient
        self._rng = None
        # parameters
        self._param = parameters
        # state monitor
        self._stateMonitor = PatientStateMonitor(parameters)
        # simulation time step
        self._delta_t = parameters.get_delta_t()

    def simulate(self, sim_length):
        """ simulate the patient over the specified simulation length """

        # random number generator for this patient
        self._rng = rndClasses.RNG(self._id)



        k = 0  # current time step

        # while the patient is alive and simulation length is not yet reached
        while self._stateMonitor.get_if_alive() and k*self._delta_t < sim_length:

            # find the transition probabilities of the future states
            trans_probs = self._param.get_transition_prob(self._stateMonitor.get_current_state())
            # create an empirical distribution
            empirical_dist = rndClasses.Empirical(trans_probs)
            # sample from the empirical distribution to get a new state
            # (returns an integer from {0, 1, 2, ...})
            new_state_index = empirical_dist.sample(self._rng)

            # update health state
            self._stateMonitor.update(k, HealthStats(new_state_index))

            # increment time step
            k += 1

    def get_survival_time(self):
        """ returns the patient's survival time"""
        return self._stateMonitor.get_survival_time()

    def get_time_to_CANCER(self):
        """ returns the patient's time to CANCER """
        return self._stateMonitor.get_time_to_CANCER()

    def get_total_discounted_cost(self):
        """ :returns total discounted cost """
        return self._stateMonitor.get_total_discounted_cost()

    def get_total_discounted_utility(self):
        """ :returns total discounted utility"""
        return self._stateMonitor.get_total_discounted_utility()

class PatientStateMonitor:
    """ to update patient outcomes (years survived, cost, etc.) throughout the simulation """
    def __init__(self, parameters):
        """
        :param parameters: patient parameters
        """
        self._currentState = parameters.get_initial_health_state() # current health state
        self._delta_t = parameters.get_delta_t()    # simulation time step
        self._survivalTime = 0          # survival time
        self._timeToCANCER = 0        # time to develop CANCER
        self._ifDevelopedCANCER = False   # if the patient developed CANCER

        # monitoring cost and utility outcomes
        self._costUtilityOutcomes = PatientCostUtilityMonitor(parameters)

    def update(self, k, next_state):
        """
        :param k: current time step
        :param next_state: next state
        """

        # if the patient has died, do nothing
        if not self.get_if_alive():
            return

        # update survival time
        if next_state in [HealthStats.CANCERDEATH, HealthStats.OTHERDEATH]:
            self._survivalTime = (k+0.5)*self._delta_t  # corrected for the half-cycle effect

        # update time until CANCER
        if ((self._currentState != HealthStats.CANCER and next_state == HealthStats.CANCER) or (self._currentState != HealthStats.MASTECTOMYCANCER and next_state == HealthStats.MASTECTOMYCANCER)):
            self._ifDevelopedCANCER = True
            self._timeToCANCER = (k + 0.5) * self._delta_t  # corrected for the half-cycle effect

        # collect cost and utility outcomes
        self._costUtilityOutcomes.update(k, self._currentState, next_state)

        # update current health state
        self._currentState = next_state

    def get_if_alive(self):
        result = True
        if self._currentState in [HealthStats.CANCERDEATH, HealthStats.OTHERDEATH]:
            result = False
        return result

    def get_current_state(self):
        return self._currentState

    def get_survival_time(self):
        """ returns the patient survival time """
        # return survival time only if the patient has died
        if not self.get_if_alive():
            return self._survivalTime
        else:
            return None

    def get_time_to_CANCER(self):
        """ returns the patient's time to CANCER """
        # return time to CANCER only if the patient has developed CANCER
        if self._ifDevelopedCANCER:
            return self._timeToCANCER
        else:
            return None

    def get_total_discounted_cost(self):
        """ :returns total discounted cost """
        return self._costUtilityOutcomes.get_total_discounted_cost()

    def get_total_discounted_utility(self):
        """ :returns total discounted utility"""
        return self._costUtilityOutcomes.get_total_discounted_utility()

class PatientCostUtilityMonitor:

    def __init__(self, parameters):

        # model parameters for this patient
        self._param = parameters

        # total cost and utility
        self._totalDiscountedCost = 0
        self._totalDiscountedUtility = 0

    def update(self, k, current_state, next_state):
        """ updates the discounted total cost and health utility
        :param k: simulation time step
        :param current_state: current health state
        :param next_state: next health state
        """

        # update cost
        cost = 0.5 * (self._param.get_annual_state_cost(current_state) +
                      self._param.get_annual_state_cost(next_state)) * self._param.get_delta_t()
        # update utility
        utility = 0.5 * (self._param.get_annual_state_utility(current_state) +
                         self._param.get_annual_state_utility(next_state)) * self._param.get_delta_t()

        # add the cost of treatment
        # if death will occur
        if next_state in [HealthStats.CANCERDEATH, HealthStats.OTHERDEATH]:
            cost += 0.5 * self._param.get_annual_treatment_cost() * self._param.get_delta_t()
        else:
            cost += 1 * self._param.get_annual_treatment_cost() * self._param.get_delta_t()

        # update total discounted cost and utility (corrected for the half-cycle effect)
        self._totalDiscountedCost += \
            EconCls.pv(cost, self._param.get_adj_discount_rate() / 2, 2*k + 1)
        self._totalDiscountedUtility += \
            EconCls.pv(utility, self._param.get_adj_discount_rate() / 2, 2*k + 1)

    def get_total_discounted_cost(self):
        """ :returns total discounted cost """
        return self._totalDiscountedCost

    def get_total_discounted_utility(self):
        """ :returns total discounted utility"""
        return  self._totalDiscountedUtility

class Cohort:
    def __init__(self, id, therapy):
        """ create a cohort of patients
        :param id: an integer to specify the seed of the random number generator
        """
        self._initial_pop_size = POP_SIZE
        self._patients = []      # list of patients

        # populate the cohort
        for i in range(self._initial_pop_size):
            # create a new patient (use id * pop_size + i as patient id)
            patient = Patient(id * self._initial_pop_size + i, ParametersFixed(therapy))
            # add the patient to the cohort
            self._patients.append(patient)

    def simulate(self):
        """ simulate the cohort of patients over the specified number of time-steps
        :returns outputs from simulating this cohort
        """
        # simulate all patients
        for patient in self._patients:
            patient.simulate(SIM_LENGTH)

        # return the cohort outputs
        return CohortOutputs(self)

    def get_initial_pop_size(self):
        return self._initial_pop_size

    def get_patients(self):
        return self._patients


class CohortOutputs:
    def __init__(self, simulated_cohort):
        """ extracts outputs from a simulated cohort
        :param simulated_cohort: a cohort after being simulated
        """

        self._survivalTimes = []        # patients' survival times
        self._times_to_CANCER = []        # patients' times to CANCER
        self._costs = []                # patients' discounted total costs
        self._utilities =[]             # patients' discounted total utilities

        # survival curve
        self._survivalCurve = \
            PathCls.SamplePathBatchUpdate('Population size over time', id, simulated_cohort.get_initial_pop_size())

        # find patients' survival times
        for patient in simulated_cohort.get_patients():


            # get the patient survival time
            survival_time = patient.get_survival_time()
            if not (survival_time is None):
                self._survivalTimes.append(survival_time)           # store the survival time of this patient
                self._survivalCurve.record(survival_time, -1)       # update the survival curve

            # get the patient's time to CANCER
            time_to_CANCER = patient.get_time_to_CANCER()
            if not (time_to_CANCER is None):
                self._times_to_CANCER.append(time_to_CANCER)

            # cost and utility
            self._costs.append(patient.get_total_discounted_cost())
            self._utilities.append(patient.get_total_discounted_utility())

        # summary statistics
        self._sumStat_survivalTime = StatCls.SummaryStat('Patient survival time', self._survivalTimes)
        self._sumState_timeToCANCER = StatCls.SummaryStat('Time until Cancer', self._times_to_CANCER)
        self._sumStat_cost = StatCls.SummaryStat('Patient discounted cost', self._costs)
        self._sumStat_utility = StatCls.SummaryStat('Patient discounted utility', self._utilities)

    def get_survival_times(self):
        return self._survivalTimes

    def get_times_to_CANCER(self):
        return self._times_to_CANCER

    def get_costs(self):
        return self._costs

    def get_utilities(self):
        return self._utilities

    def get_sumStat_survival_times(self):
        return self._sumStat_survivalTime

    def get_sumStat_time_to_CANCER(self):
        return self._sumState_timeToCANCER

    def get_sumStat_discounted_cost(self):
        return self._sumStat_cost

    def get_sumStat_discounted_utility(self):
        return self._sumStat_utility

    def get_survival_curve(self):
        return self._survivalCurve

# Markov support functions, based on SupportMarkovModel.py
def print_outcomes(simOutput, therapy_name):
    """ prints the outcomes of a simulated cohort
    :param simOutput: output of a simulated cohort
    :param therapy_name: the name of the selected therapy
    """
    # mean and confidence interval text of patient survival time
    survival_mean_CI_text = F.format_estimate_interval(
        estimate=simOutput.get_sumStat_survival_times().get_mean(),
        interval=simOutput.get_sumStat_survival_times().get_t_CI(alpha=ALPHA),
        deci=2)

    # mean and confidence interval text of time to CANCER
    time_to_CANCER_CI_text = F.format_estimate_interval(
        estimate=simOutput.get_sumStat_time_to_CANCER().get_mean(),
        interval=simOutput.get_sumStat_time_to_CANCER().get_t_CI(alpha=ALPHA),
        deci=2)

    # mean and confidence interval text of discounted total cost
    cost_mean_CI_text = F.format_estimate_interval(
        estimate=simOutput.get_sumStat_discounted_cost().get_mean(),
        interval=simOutput.get_sumStat_discounted_cost().get_t_CI(alpha=ALPHA),
        deci=0,
        form=F.FormatNumber.CURRENCY)

    # mean and confidence interval text of discounted total utility
    utility_mean_CI_text = F.format_estimate_interval(
        estimate=simOutput.get_sumStat_discounted_utility().get_mean(),
        interval=simOutput.get_sumStat_discounted_utility().get_t_CI(alpha=ALPHA),
        deci=2)

    # print outcomes
    print(therapy_name)
    print("  Estimate of mean survival time and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          survival_mean_CI_text)
    print("  Estimate of mean time to cancer and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          time_to_CANCER_CI_text)
    print("  Estimate of discounted cost and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          cost_mean_CI_text)
    print("  Estimate of discounted utility and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          utility_mean_CI_text)
    print("")

def draw_survival_curves_and_histograms(simOutputs_standard, simOutputs_population):
    """ draws the survival curves and the histograms of time until cancer
    :param simOutputs_standard: output of a cohort simulated under standard testing
    :param simOutputs_population: output of a cohort simulated under population testing
    """
    # get survival curves of both treatments
    survival_curves = [
        simOutputs_standard.get_survival_curve(),
        simOutputs_population.get_survival_curve()
    ]

    # graph survival curve
    PathCls.graph_sample_paths(
        sample_paths=survival_curves,
        title='Survival curve',
        x_label='Simulation time step (year)',
        y_label='Number of alive patients',
        legends=['Standard Testing', 'Population Testing']
    )

    # histograms of survival times
    set_of_survival_times = [
        simOutputs_standard.get_survival_times(),
        simOutputs_population.get_survival_times()
    ]

    # graph histograms
    Figs.graph_histograms(
        data_sets=set_of_survival_times,
        title='Histogram of patient survival time',
        x_label='Survival time (year)',
        y_label='Counts',
        bin_width=1,
        legend=['Mono Therapy', 'Combination Therapy'],
        transparency=0.6
    )

def print_comparative_outcomes(simOutputs_standard, simOutputs_population):
    """ prints average increase in survival time, discounted cost, and discounted utility
    under population testing compared to standard testing
    :param simOutputs_standard: output of a cohort simulated under standard therapy
    :param simOutputs_population: output of a cohort simulated under population
    """

    # increase in survival time under population testing with respect to standard testing
    increase_survival_time = Stat.DifferenceStatIndp(
            name='Increase in survival time',
            x=simOutputs_population.get_survival_times(),
            y_ref=simOutputs_standard.get_survival_times())

    # estimate and CI
    estimate_CI = F.format_estimate_interval(
        estimate=increase_survival_time.get_mean(),
        interval=increase_survival_time.get_t_CI(alpha=ALPHA),
        deci=2)
    print("Average increase in survival time "
          "and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          estimate_CI)

    # increase in discounted total cost under population testing with respect to standard testing
    increase_discounted_cost = Stat.DifferenceStatIndp(
            name='Increase in discounted cost',
            x=simOutputs_population.get_costs(),
            y_ref=simOutputs_standard.get_costs())

        # estimate and CI
    estimate_CI = F.format_estimate_interval(
        estimate=increase_discounted_cost.get_mean(),
        interval=increase_discounted_cost.get_t_CI(alpha=ALPHA),
        deci=0,
        form=F.FormatNumber.CURRENCY)
    print("Average increase in discounted cost "
          "and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          estimate_CI)

    # increase in discounted total utility under combination therapy with respect to mono therapy
    increase_discounted_utility = Stat.DifferenceStatIndp(
            name='Increase in discounted cost',
            x=simOutputs_population.get_utilities(),
            y_ref=simOutputs_standard.get_utilities())

        # estimate and CI
    estimate_CI = F.format_estimate_interval(
        estimate=increase_discounted_utility.get_mean(),
        interval=increase_discounted_utility.get_t_CI(alpha=ALPHA),
        deci=2)
    print("Average increase in discounted utility "
          "and {:.{prec}%} confidence interval:".format(1 - ALPHA, prec=0),
          estimate_CI)

def report_CEA_CBA(simOutputs_standard, simOutputs_population):
    """ performs cost-effectiveness analysis
    :param simOutputs_standard: output of a cohort simulated under standard testing
    :param simOutputs_population: output of a cohort simulated under population testing
    """
    # define two strategies
    standard_testing_strategy = Econ.Strategy(
        name='Standard Testing',
        cost_obs=simOutputs_standard.get_costs(),
        effect_obs=simOutputs_standard.get_utilities()
    )
    population_testing_strategy = Econ.Strategy(
        name='Population Testing',
        cost_obs=simOutputs_population.get_costs(),
        effect_obs=simOutputs_population.get_utilities()
    )

    # CEA
    CEA = Econ.CEA(
            strategies=[standard_testing_strategy, population_testing_strategy],
            if_paired=False
        )

    # show the CE plane
    CEA.show_CE_plane(
        title='Cost-Effectiveness Analysis',
        x_label='Additional discounted utility',
        y_label='Additional discounted cost',
        show_names=True,
        show_clouds=True,
        show_legend=True,
        figure_size=6,
        transparency=0.3
    )

    # report the CE table
    CEA.build_CE_table(
        interval=Econ.Interval.CONFIDENCE,
        alpha=ALPHA,
        cost_digits=0,
        effect_digits=2,
        icer_digits=0,
    )

    # CBA
    NBA = Econ.CBA(
            strategies=[standard_testing_strategy, population_testing_strategy],
            if_paired=False
        )

    # show the net monetary benefit figure
    NBA.graph_deltaNMB_lines(
        min_wtp=0,
        max_wtp=150000,
        title='Cost-Benefit Analysis',
        x_label='Willingness-to-pay for one additional QALY ($)',
        y_label='Incremental Net Monetary Benefit ($)',
        interval=Econ.Interval.CONFIDENCE,
        show_legend=True,
        figure_size=6
    )

# Data
# simulation settings
POP_SIZE = 2500     # cohort population size
SIM_LENGTH = 60    # length of simulation (years)
ALPHA = 0.05        # significance level for calculating confidence intervals
DELTA_T = 1/12     # years
DISCOUNT = 0.035     # annual discount rate
TEST_COST = 400 # cost of genetic test

ANNUAL_STATE_COST = [
    1534,   # WELL
    13504.5,   # CANCER
    0,   # CANCERDEATH
    12596,  # MASTECTOMY
    1534, # POSTMASTECTOMY
    8314.65, # POSTMASTECTOMYCANCER
    0, #OTHERDEATH
    ]

ANNUAL_STATE_UTILITY = [
    1.0,   # WELL
    0.655,   # CANCER
    0,   # CANCERDEATH
    0.85,  # MASTECTOMY
    0.95, # POSTMESTECTOMY
    0.73, # POSTMASTECTOMYCANCER
    0, #OTHERDEATH
    ]


mast_rate = -numpy.log(1-(numpy.random.beta(105, 273)))
sta_mast_rate = 0.0015*mast_rate
pop_mast_rate = 0.008*mast_rate

print(sta_mast_rate)
print(pop_mast_rate)

no_brca_transition = [
    [0,  0.001260794,    0,    0, 0, 0, 0.017951162],   # WELL
    [0.439765016,     0,    0.00209022,    0, 0, 0, 0.017951162],   # CANCER
    [0,     0,     0,   0, 0, 0, 0],   # CANCERDEATH
    [0,     0,      0,   0, 0, 0, 0],   # MASTECTOMY
    [0,     0,      0,   0, 0, 0, 0],   # POSTMASTECTOMY
    [0,     0,      0,   0, 0, 0, 0],   # MASTECTOMYCANCER
    [0,     0,      0,   0, 0, 0, 0] #OTHERDEATH
    ]

untested_brca_transition = [
    [0,  0.007117385,    0,    0, 0, 0, 0.017951162],   # WELL
    [0.439765016,     0,    0.00209022,    0, 0, 0, 0.017951162],   # CANCER
    [0,     0,     0,   0, 0, 0, 0],   # CANCERDEATH
    [0,     0,      0,   0, 0, 0, 0],   # MASTECTOMY
    [0,     0,      0,   0, 0, 0, 0],   # POSTMASTECTOMY
    [0,     0,      0,   0, 0, 0, 0],   # MASTECTOMYCANCER
    [0,     0,      0,   0, 0, 0, 0] #OTHERDEATH
    ]

tested_brca_transition = [
    [0,  0.007117385,    0,    mast_rate, 0, 0, 0.017951162],   # WELL
    [0.439765016,     0,    0.00209022,    0, 0, 0, 0.017951162],   # CANCER
    [0,     0,     0,   0, 0, 0, 0],   # CANCERDEATH
    [0,     0,      0,   0, 12, 0, 0],   # MASTECTOMY
    [0,     0,      0,   0, 0, 0.00006509, 0.017951162],   # POSTMASTECTOMY
    [0,     0,      0.000010451,   0, 0.439765016, 0, 0.017951162],   # MASTECTOMYCANCER
    [0,     0,      0,   0, 0, 0, 0] #OTHERDEATH
    ]

population_transition = [
    [0,  0.001106522,    0,    pop_mast_rate, 0, 0, 0.017951162],   # WELL
    [0.439765016,     0,    0.00209022,    0, 0, 0, 0.017951162],   # CANCER
    [0,     0,     0,   0, 0, 0, 0],   # CANCERDEATH
    [0,     0,      0,   0, 12, 0, 0],   # MASTECTOMY
    [0,     0,      0,   0, 0, 0.00006509, 0.017951162],   # POSTMASTECTOMY
    [0,     0,      0.000010451,   0, 0.439765016, 0, 0.017951162],   # MASTECTOMYCANCER
    [0,     0,      0,   0, 0, 0, 0] #OTHERDEATH
    ]

standard_transition = [
    [0,  0.00130179,    0,    sta_mast_rate, 0, 0, 0.017951162],   # WELL
    [0.439765016,     0,    0.00209022,    0, 0, 0, 0.017951162],   # CANCER
    [0,     0,     0,   0, 0, 0, 0],   # CANCERDEATH
    [0,     0,      0,   0, 12, 0, 0],   # MASTECTOMY
    [0,     0,      0,   0, 0, 0.00006509, 0.017951162],   # POSTMASTECTOMY
    [0,     0,      0.000010451,   0, 0.439765016, 0, 0.017951162],   # MASTECTOMYCANCER
    [0,     0,      0,   0, 0, 0, 0] #OTHERDEATH
    ]

# Run
# create cohorts
standard = Cohort(
    id=1,
    therapy=Therapies.STANDARD)

population = Cohort(
    id=1,
    therapy=Therapies.POPULATION)

# simulate the cohorts
standardOutputs = standard.simulate()
populationOutputs = population.simulate()

# graph survival curve
PathCls.graph_sample_path(
    sample_path=standardOutputs.get_survival_curve(),
    title='Survival curve for standard testing',
    x_label='Simulation time step',
    y_label='Number of alive patients'
    )

# graph histogram of survival times
Figs.graph_histogram(
    data=standardOutputs.get_survival_times(),
    title='Survival times of patients for standard testing',
    x_label='Survival time (years)',
    y_label='Counts',
    bin_width=1
)
# graph survival curve
PathCls.graph_sample_path(
    sample_path=populationOutputs.get_survival_curve(),
    title='Survival curve for population testing',
    x_label='Simulation time step',
    y_label='Number of alive patients'
    )

# graph histogram of survival times
Figs.graph_histogram(
    data=populationOutputs.get_survival_times(),
    title='Survival times of patients for population testing',
    x_label='Survival time (years)',
    y_label='Counts',
    bin_width=1
)


# print the estimates for the mean survival time and mean time to cancer
print_outcomes(standardOutputs, 'Standard testing:')

print_outcomes(populationOutputs, 'Population testing:')

# draw survival curves and histograms
draw_survival_curves_and_histograms(standardOutputs, populationOutputs)

# print comparative outcomes
print_comparative_outcomes(standardOutputs, populationOutputs)

# report the CEA results
report_CEA_CBA(standardOutputs, populationOutputs)



