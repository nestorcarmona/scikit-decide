import math
from argparse import Action
from enum import Enum

# data and math
from math import asin, atan2, cos, radians, sin, sqrt
import pandas as pd
import numpy as np
import networkx as nx

# typing
from typing import Any, Callable, Optional, Tuple, Union, Dict, List

# plotting
import matplotlib.pyplot as plt

# aircraft performance model
from openap.extra.aero import bearing as aero_bearing
from openap.extra.aero import distance, ft, kts, latlon, mach2tas, mach2cas, nm, mach2cas, cas2tas
from openap.extra.nav import airport
from openap.prop import aircraft
from pygeodesy.ellipsoidalVincenty import LatLon

from skdecide import DeterministicPlanningDomain, ImplicitSpace, Solver, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions

# custom aircraft performance model
from skdecide.hub.domain.flight_planning.aircraft_performance.base import (
    AircraftPerformanceModel,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import (
    load_aircraft_engine_params,
)
from skdecide.hub.domain.flight_planning.flightplanning_utils import (
    plot_full,
    plot_trajectory,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.get_weather_noaa import (
    get_weather_matrix,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.GenericInterpolator import (
    GenericWindInterpolator,
)
from skdecide.hub.space.gym import EnumSpace, ListSpace, TupleSpace, DiscreteSpace
from skdecide.utils import load_registered_solver

try:
    from IPython.display import clear_output as ipython_clear_output
except ImportError:
    ipython_available = False
else:
    ipython_available = True


def clear_output(wait=True):
    if ipython_available:
        ipython_clear_output(wait=wait)


class WeatherDate:
    day: int
    month: int
    year: int
    forecast: str
    leapyear: bool

    def __init__(self, day, month, year, forecast="nowcast") -> None:
        self.day = int(day)
        self.month = int(month)
        self.year = int(year)
        self.forecast = forecast
        self.leapyear = self.year % 400 == 0 or (
            self.year % 100 != 0 and self.year % 4 == 0
        )

    def __hash__(self) -> int:
        return hash((self.day, self.month, self.year, self.forecast))

    def __eq__(self, other: object) -> bool:
        return (
            self.day == other.day
            and self.month == other.month
            and self.year == other.year
            and self.forecast == other.forecast
        )

    def __ne__(self, other: object) -> bool:
        return (
            self.day != other.day
            or self.month != other.month
            or self.year != other.year
            or self.forecast != other.forecast
        )

    def __str__(self) -> str:
        day = str(self.day)
        month = str(self.month)

        if len(day) == 1:
            day = "0" + day
        if len(month) == 1:
            month = "0" + month

        return f"[{day} {month} {self.year}, forecast : {self.forecast}]"

    def to_dict(self) -> dict:
        day = str(self.day)
        month = str(self.month)

        if len(day) == 1:
            day = "0" + day
        if len(month) == 1:
            month = "0" + month

        return {
            "year": str(self.year),
            "month": str(month),
            "day": str(day),
            "forecast": self.forecast,
        }

    def next_day(self):
        day = self.day
        month = self.month
        year = self.year
        if month == 12 and day == 31:
            year += 1
            month = 1
            day = 1

        elif month in (1, 3, 5, 7, 8, 10) and day == 31:
            day = 1
            month += 1

        elif month in (4, 6, 9, 11) and day == 30:
            day = 1
            month += 1

        elif month == 2:
            if (self.leap_year and day == 29) or (not (self.leap_year) and day == 28):
                day = 1
                month = 3
            else:
                day += 1

        else:
            day += 1

        return WeatherDate(day, month, year, forecast=self.forecast)

    def previous_day(self):
        day = self.day
        month = self.month
        year = self.year
        if month == 1 and day == 1:
            year -= 1
            month = 12
            day = 31

        elif month in (5, 7, 10, 12) and day == 1:
            day = 30
            month -= 1

        elif month in (2, 4, 6, 8, 9, 11) and day == 1:
            day = 31
            month -= 1

        elif month == 3 and day == 1:
            if self.leap_year:
                day = 29
                month = 2
            else:
                day = 28
                month = 2

        else:
            day -= 1

        return WeatherDate(day, month, year, forecast=self.forecast)


class State:
    """
    Definition of a aircraft state during the flight plan
    """

    trajectory: pd.DataFrame
    id: Dict[str, float]

    def __init__(self, trajectory, id):
        """Initialisation of a state

        Args:
            trajectory : Trajectory information of the flight
            id: Node id in the airway graph
        """
        self.trajectory = trajectory
        self.id = id

        if trajectory is not None:
            self.mass = trajectory.iloc[-1]["mass"]
            self.alt = trajectory.iloc[-1]["alt"]
            self.time = trajectory.iloc[-1]["ts"]
        else:
            self.mass = None
            self.alt = None
            self.time = None

    def __hash__(self):
        return hash((self.id, int(self.mass), self.alt, int(self.time)))

    def __eq__(self, other):
        return (
            self.id == other.id
            and int(self.mass) == int(other.mass)
            and self.alt == other.alt
            and int(self.time) == int(other.time)
        )

    def __ne__(self, other):
        return (
            self.id != other.id
            or int(self.mass) != int(other.mass)
            or self.alt != other.alt
            or int(self.time) != int(other.time)
        )

    def __str__(self):
        return f"[{self.trajectory.iloc[-1]['ts']:.2f} \
            {self.id} \
            {self.trajectory.iloc[-1]['alt']:.2f} \
            {self.trajectory.iloc[-1]['mass']:.2f}]"


class H_Action(Enum):
    """
    Horizontal action that can be perform by the aircraft
    """

    left = -1
    straight = 0
    right = 1


class V_Action(Enum):
    """
    Vertical action that can be perform by the aircraft
    """

    climb = 1
    cruise = 0
    descent = -1


class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent


class FlightPlanningDomain(
    DeterministicPlanningDomain, UnrestrictedActions, Renderable
):
    """Automated flight planning domain.

    Domain definition
    -----------------

    The flight planning domain can be quickly defined as:

    - An origin, as ICAO code of an airport,
    - A destination, as ICAO code of an airport,
    - An aircraft type, as a string recognizable by the OpenAP library.

    Airways graph
    -------------

    A three-dimensional airway graph of waypoints is created. The graph is following the great circle
    which represents the shortest path between the origin and the destination.
    The planner computes a plan by choosing waypoints in the graph, which are represented by 4-dimensionnal states.
    There are 3 phases in the graph:

    - The climbing phase: represented by a dictionnary of 3 values (below_10k_ft, from_10k_to_crossover, above_crossover) which are the CAS or Mach number.
    - The cruise phase: represented by a dictionnary of 1 value (above_crossover) which is the Mach number.
    - The descent phase: represented by a dictionnary of 3 values (below_10k_ft, from_10k_to_crossover, above_crossover) which are the CAS or Mach number.
    
    For the climbing and descent phases, a rate of climb and descent is also defined.

    The flight planning domain allows to choose a number of branches, which represent the lateral waypoints in the graph, 
    and a number of steps for each phase. It is also possible to choose different width (tiny, small, normal, large, xlarge) which will increase
    or decrease the graph width.

    State representation
    --------------------

    Here, the states are represented by 4 features:

    - The position in the graph (x,y,z)
    - The aircraft mass, which can also represent the fuel consumption (integer)
    - The altitude (float)
    - The time (seconds)

    Wind interpolation
    ------------------

    The flight planning domain can take in consideration the wind conditions.
    That interpolation have a major impact on the results, as jet streams are high altitude wind
    which can increase or decrease the ground speed of the aircraft.
    It also have an impact on the computation time of a flight plan,
    as the objective and heuristic function became more complex.

    Objective (or cost) functions
    -----------------------------

    There is three possible objective functions:

    - Fuel (Default)
    - Distance
    - Time

    The chosen objective will represent the cost to go from a state to another. The aim of the algorithm is to minimize the cost.

    Heuristic functions
    -------------------

    When using an A* algorithm to compute the flight plan, we need to feed it with a heuristic function, which guide the algorithm.
    For now, there is 5 different (not admissible) heuristic function, depending on `self.heuristic_name`:

    - fuel, which computes the required fuel to get to the goal. It takes in consideration the local wind & speed of the aircraft.
    - time, which computes the required time to get to the goal. It takes in consideration the local wind & speed of the aircraft.
    - distance, wich computes the distance to the goal.
    - lazy_fuel, which propagates the fuel consumed so far.
    - lazy_time, which propagates the time spent on the flight so far
    - None : we give a 0 cost value, which will transform the A* algorithm into a Dijkstra-like algorithm.

    Aircraft performance models
    --------------------------

    The flight planning domain can use two possible A/C performance models:

    - OpenAP: the aircraft performance model is based on the OpenAP library.
    - Poll-Schumann: the aircraft performance model is based on Poll-Schumann equations as stated on the paper: "An estimation
    method for the fuel burn and other performance characteristics of civil transport aircraft in the cruise" by Poll and Schumann;
    The Aeronautical Journal, 2020.

    Optional features
    -----------------

    The flight planning domain has several optional features:

    - Fuel loop: this is an optimisation of the loaded fuel for the aircraft.
      It will run some flights to computes the loaded fuel, using distance objective & heuristic.

    - Constraints definition: you can define constraints such as

        - A time constraint, represented by a time windows
        - A fuel constraint, represented by the maximum fuel for instance.

    - Slopes: you can define your own climbing & descending slopes which have to be between 10.0 and 25.0.

    """

    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Tuple[H_Action, V_Action]  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent

    def __init__(
        self,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        actype: str,
        weather_date: WeatherDate,
        wind_interpolator: Optional[GenericWindInterpolator] = None,
        objective: str = "fuel",
        heuristic_name: str = "fuel",
        perf_model_name: str = "openap",
        constraints: Optional[Dict[str, float]] = None,
        climb_profile: Dict[str, float] = {"below_10k_ft": 250., "from_10k_to_crossover": 270., "above_crossover": 0.78},
        cruise_profile: Dict[str, float] = {"above_crossover": 0.83},
        descent_profile: Dict[str, float] = {"below_10k_ft": 280., "from_10k_to_crossover": 280., "above_crossover": 0.8},
        rate_of_climb_descent: Dict[str, float] = {"climb": 1_500.0, "descent": 2_000.0},
        steps: Dict[str, int] = {"n_steps_climb": 5, "n_steps_cruise": 5, "n_steps_cruise_climb": 10, "n_steps_descent": 5},
        n_branches: int = 3,
        plane_heading: float = 0,
        take_off_weight: Optional[int] = None,
        fuel_loaded: Optional[float] = None,
        fuel_loop: bool = False,
        fuel_loop_solver_factory: Optional[Callable[[], Solver]] = None,
        fuel_loop_tol: float = 1e-3,
        graph_width: str = "medium",
        res_img_dir: Optional[str] = None,
        starting_time: float = 3_600.0 * 8.0,
    ):
        """Initialisation of a flight planning instance

        Args:
            origin (Union[str, tuple]):
                ICAO code of the airport, or a tuple (lat,lon,alt), of the origin of the flight plan. Altitude should be in ft
            destination (Union[str, tuple]):
                ICAO code of the airport, or a tuple (lat,lon,alt), of the destination of the flight plan. Altitude should be in ft
            actype (str):
                Aircarft type describe in openap datas (https://github.com/junzis/openap/tree/master/openap/data/aircraft)
            weather_date (WeatherDate, optional):
                Date for the weather, needed for days management.
                If None, no wind will be applied.
            wind_interpolator (GenericWindInterpolator, optional):
                Wind interpolator for the flight plan. If None, create one from the specified weather_date.
                The data is either already present locally or be downloaded from https://www.ncei.noaa.gov
            objective (str, optional):
                Cost function of the flight plan. It can be either fuel, distance or time. Defaults to "fuel".
            heuristic_name (str, optional):
                Heuristic of the flight plan, it will guide the aircraft through the graph. It can be either fuel, distance or time. Defaults to "fuel".
            perf_model_name (str, optional):
                Aircraft performance model used in the flight plan. It can be either openap or PS (Poll-Schumann). Defaults to "openap".
            constraints (_type_, optional):
                Constraints dictionnary (keyValues : ['time', 'fuel'] ) to be defined in for the flight plan. Defaults to None.
            climb_profile (Dict[str, float], optional):
                Climbing profile of the aircraft. Defaults to {"below_10k_ft": 250., "from_10k_to_crossover": 270., "above_crossover": 0.78}.
            cruise_profile (Dict[str, float], optional):
                Cruise profile of the aircraft. Defaults to {"above_crossover": 0.83}.
            descent_profile (Dict[str, float], optional):
                Descent profile of the aircraft. Defaults to {"below_10k_ft": 280., "from_10k_to_crossover": 280., "above_crossover": 0.8}.
            rate_of_climb_descent (Dict[str, float], optional):
                Rate of climb and descent of the aircraft. Defaults to {"climb": 1_500.0, "descent": 2_000.0}.
            steps (Dict[str, int], optional):
                Number of steps for each phase of the flight plan. Defaults to {"n_steps_climb": 5, "n_steps_cruise": 5, "n_steps_cruise_climb": 10, "n_steps_descent": 5}.
            n_branches (int, optional):
                Number of branches in the graph. Defaults to 3.
            plane_heading (float, optional):
                Heading of the aircraft. Defaults to 0.
            take_off_weight (int, optional):
                Take off weight of the aircraft. Defaults to None.
            fuel_loaded (float, optional):
                Fuel loaded in the aricraft for the flight plan. Defaults to None.
            fuel_loop (bool, optional):
                Boolean to create a fuel loop to optimize the fuel loaded for the flight. Defaults to False
            fuel_loop_solver_factory (solver factory, optional):
                Solver factory used in the fuel loop. Defaults to LazyAstar.
            graph_width (str, optional):
                Airways graph width, in ["small", "medium", "large", "xlarge"]. Defaults to medium
            res_img_dir (str, optional):
                Directory in which images will be saved. Defaults to None
            starting_time (float, optional):
                Start time of the flight, in seconds. Defaults to 8AM (3_600.0 * 8.0)
        """

        # Initialisation of the origin and the destination
        self.origin, self.destination = origin, destination
        if isinstance(origin, str):  # Origin is an airport
            ap1 = airport(origin)
            self.lat1, self.lon1, self.alt1 = ap1["lat"], ap1["lon"], ap1["alt"] # type: ignore
        else:  # Origin is geographic coordinates
            self.lat1, self.lon1, self.alt1 = origin

        if isinstance(destination, str):  # Destination is an airport
            ap2 = airport(destination)
            self.lat2, self.lon2, self.alt2 = ap2["lat"], ap2["lon"], ap2["alt"] # type: ignore
        else:  # Destination is geographic coordinates
            self.lat2, self.lon2, self.alt2 = destination

        self.start_time = starting_time
        # Retrieve the aircraft datas in openap library
        self.actype = actype
        self.ac = aircraft(actype)

        # Initialisation of the objective & heuristic, the constraints and the wind interpolator
        if heuristic_name in (
            "distance",
            "fuel",
            "lazy_fuel",
            "time",
            "lazy_time",
            None,
        ):
            self.heuristic_name = heuristic_name
        else:
            self.heuristic_name = "fuel"

        if objective in ("distance", "fuel", "time"):
            self.objective = objective
        else:
            self.objective = "fuel"
        self.constraints = constraints

        self.weather_date = weather_date
        self.initial_date = weather_date

        if wind_interpolator is None:
            self.weather_interpolator = self.get_weather_interpolator()

        # Initialisation of the aircraft performance model
        self.perf_model = AircraftPerformanceModel(actype, perf_model_name)
        self.perf_model_name = perf_model_name
        
        # Initialisation of the flight phases and the graph
        self.alt_crossover = self.perf_model.compute_crossover_altitude(climb_profile["from_10k_to_crossover"], climb_profile["above_crossover"])
        self.climb_profile = climb_profile
        self.cruise_profile = cruise_profile
        self.descent_profile = descent_profile

        p0 = LatLon(self.lat1, self.lon1, self.alt1 * ft)
        p1 = LatLon(self.lat2, self.lon2, self.alt2 * ft)
        # TODO: remove this line
        plane_heading = aero_bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        self.network = self.set_network(
            p0=p0,  # alt ft -> meters
            p1=p1,  # alt ft -> meters
            climb_profile=self.climb_profile,
            cruise_profile=self.cruise_profile,
            descent_profile=self.descent_profile,
            rate_of_climb_descent=rate_of_climb_descent,
            plane_heading=plane_heading,

            steps=steps,
            n_branches=n_branches,

            graph_width=graph_width,
        )

        self.fuel_loaded = fuel_loaded

        # Initialisation of the flight plan, with the initial state
        if fuel_loop:
            if fuel_loop_solver_factory is None:
                LazyAstar = load_registered_solver("LazyAstar")
                fuel_loop_solver_factory = lambda: LazyAstar(
                    heuristic=lambda d, s: d.heuristic(s)
                )
            fuel_loaded = fuel_optimisation(
                origin=origin,
                destination=destination,
                actype=self.actype,
                constraints=constraints,
                weather_date=weather_date,
                solver_factory=fuel_loop_solver_factory,
                fuel_tol=fuel_loop_tol,
            )
            # Adding fuel reserve (but we can't put more fuel than maxFuel)
            fuel_loaded = min(1.1 * fuel_loaded, self.ac["limits"]["MFC"])
        elif fuel_loaded:
            self.constraints["fuel"] = (
                0.97 * fuel_loaded
            )  # Update of the maximum fuel there is to be used
        else:
            fuel_loaded = self.ac["limits"]["MFC"]

        self.fuel_loaded = fuel_loaded

        assert (
            fuel_loaded <= self.ac["limits"]["MFC"]
        )  # Ensure fuel loaded <= fuel capacity

        aircraft_params = load_aircraft_engine_params(actype)

        self.start = State(
            trajectory=pd.DataFrame(
                [
                    {
                        "ts": self.start_time,
                        "lat": self.lat1,
                        "lon": self.lon1,
                        "mass": aircraft_params["amass_mtow"]
                        if take_off_weight is None
                        else take_off_weight
                        - 0.8
                        * (
                            self.ac["limits"]["MFC"] - self.fuel_loaded
                        ),  # Here we compute the weight difference between the fuel loaded and the fuel capacity
                        "cas_mach": climb_profile["below_10k_ft"],
                        "fuel": 0.0,
                        "alt": self.alt1,
                    }
                ]
            ),
            id=0)

        self.res_img_dir = res_img_dir

    # Class functions
    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """Compute the next state

        Args:
            memory (D.T_state): The current state
            action (D.T_event): The action to perform

        Returns:
            D.T_state: The next state
        """

        trajectory = memory.trajectory.copy()

        # Get current node information
        current_node_id = memory.id
        current_height = self.network.nodes[current_node_id]["height"]
        current_phase = self.network.nodes[current_node_id]["phase"]
        current_branch_id = self.network.nodes[current_node_id]["branch_id"]

        # Get successors information
        node_successors = list(self.network.successors(current_node_id))
        successors_heights = np.array([self.network.nodes[node_id]["height"] for node_id in node_successors])
        successors_branch_ids = np.array([self.network.nodes[node_id]["branch_id"] for node_id in node_successors])

        # Horizontal actions
        if action[0] == H_Action.straight:
            index_headings = np.where(successors_branch_ids == current_branch_id)[0]
        elif action[0] == H_Action.right:
            index_headings = np.where(successors_branch_ids > current_branch_id)[0]
        elif action[0] == H_Action.left:
            index_headings = np.where(successors_branch_ids < current_branch_id)[0]

        # Vertical actions
        if action[1] == V_Action.cruise:
            index_heights = np.where(successors_heights == current_height)[0]
        elif action[1] == V_Action.climb:
            index_heights = np.where(successors_heights > current_height)[0]
        elif action[1] == V_Action.descent:
            index_heights = np.where(successors_heights < current_height)[0]

        if len(index_headings) == 0 or len(index_heights) == 0:
            return memory

        # Compute the intersection of the indexes to get the next node to reach
        index = np.intersect1d(index_headings, index_heights)

        if len(index) == 0:
            print("There seem to be an issue with the index intersection.")
            return memory
        else:
            index = index[0]

        # Get the next node information
        next_node = node_successors[index]
        to_lat = self.network.nodes[next_node]["lat"]
        to_lon = self.network.nodes[next_node]["lon"]
        to_alt = self.network.nodes[next_node]["height"] / ft

        # Compute the next trajectory
        trajectory = self.flying(trajectory.tail(1), (to_lat, to_lon, to_alt), current_phase)

        # Update the next state
        state = State(
            pd.concat([memory.trajectory, trajectory], ignore_index=True),
            next_node,
        )
        return state

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: D.T_state,
    ) -> Value[D.T_value]:
        """
        Get the value (reward or cost) of a transition.
        Set cost to distance travelled between points

        Args:
            memory (D.T_state): The current state
            action (D.T_event): The action to perform
            next_state (Optional[D.T_state], optional): The next state. Defaults to None.

        Returns:
            Value[D.T_value]: Cost to go from memory to next state
        """
        assert memory != next_state, "Next state is the same as the current state"
        if self.objective == "distance":
            cost = LatLon.distanceTo(
                LatLon(
                    memory.trajectory.iloc[-1]["lat"],
                    memory.trajectory.iloc[-1]["lon"],
                    memory.trajectory.iloc[-1]["alt"] * ft,
                ),
                LatLon(
                    next_state.trajectory.iloc[-1]["lat"],
                    next_state.trajectory.iloc[-1]["lon"],
                    next_state.trajectory.iloc[-1]["alt"] * ft,
                ),
            )
        elif self.objective == "time" or self.objective == "lazy_time":
            cost = (
                next_state.trajectory.iloc[-1]["ts"] - memory.trajectory.iloc[-1]["ts"]
            )
        else:
            cost = (
                memory.trajectory.iloc[-1]["mass"]
                - next_state.trajectory.iloc[-1]["mass"]
            )
        return Value(cost=cost)

    def _get_initial_state_(self) -> D.T_state:
        """
        Get the initial state.

        Set the start position as initial state.
        """
        return self.start

    def _get_goals_(self) -> Space[D.T_observation]:
        """
        Get the domain goals space (finite or infinite set).

        Set the end position as goal.
        """
        return ImplicitSpace(lambda x: len(list(self.network.successors(x.id))) == 0)

    def _get_terminal_state_time_fuel(self, state: State) -> dict:
        """
        Get the domain terminal state information to compare with the constraints

        Args:
            state (State): terminal state to retrieve the information on fuel and time.

        Returns:
            dict: dictionnary containing both fuel and time information.
        """
        fuel = 0.0
        for trajectory in state.trajectory.iloc:
            fuel += trajectory["fuel"]

        if (
            state.trajectory.iloc[-1]["ts"] < self.start_time
        ):  # The flight arrives the next day
            time = 3_600 * 24 - self.start_time + state.trajectory.iloc[-1]["ts"]
        else:
            time = state.trajectory.iloc[-1]["ts"] - self.start_time

        return {"time": time, "fuel": fuel}
    
    def _is_terminal(self, state: State) -> D.T_predicate:
        """
        Indicate whether a state is terminal.

        Stop an episode only when goal reached.
        """
        current_node_id = state.id
        # The state is terminal if it does not have any successors
        return  len(list(self.network.successors(current_node_id))) == 0

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        """
        Get the applicable actions from a state.
        """
        # Get current node information
        current_node_id = memory.id
        current_height = self.network.nodes[current_node_id]["height"]
        current_branch_id = self.network.nodes[current_node_id]["branch_id"]

        # Get successors information
        node_successors = list(self.network.successors(current_node_id))
        successors_heights = np.array([self.network.nodes[node_id]["height"] for node_id in node_successors])
        successors_branch_ids = np.array([self.network.nodes[node_id]["branch_id"] for node_id in node_successors])

        # V_Action
        index_climb = (np.where(successors_heights > current_height)[0], V_Action.climb)
        index_descend = (np.where(successors_heights < current_height)[0], V_Action.descent)
        index_cruise = (np.where(successors_heights == current_height)[0], V_Action.cruise)

        # H_Action
        index_straight = (np.where(successors_branch_ids == current_branch_id)[0], H_Action.straight)
        index_down = (np.where(successors_branch_ids < current_branch_id)[0], H_Action.left)
        index_up = (np.where(successors_branch_ids > current_branch_id)[0], H_Action.right)

        space = []

        for v_actions in [index_climb, index_descend, index_cruise]:
            for h_actions in [index_straight, index_down, index_up]:
                if len(v_actions[0]) > 0 and len(h_actions[0]) > 0:
                    # Compute intersection of the indexes
                    index = np.intersect1d(v_actions[0], h_actions[0])
                    if len(index) > 0:
                        space.append((h_actions[1], v_actions[1]))

        return ListSpace(space)

    def _get_action_space_(self) -> Space[D.T_event]:
        """
        Define action space.
        """
        return EnumSpace((H_Action, V_Action)) # type: ignore

    def _get_observation_space_(self) -> Space[D.T_observation]:
        """
        Define observation space.
        """
        return TupleSpace((
                DiscreteSpace(self.network.number_of_nodes()), # type: ignore
                DiscreteSpace(self.network.number_of_edges()))
            ) 

    def _render_from(self, memory: State, **kwargs: Any) -> Any:
        """
        Render visually the map.

        Returns:
            matplotlib figure
        """
        return plot_trajectory(
            self.lat1,
            self.lon1,
            self.lat2,
            self.lon2,
            memory.trajectory,
        )

    def heuristic(self, s: D.T_state, heuristic_name: Optional[str] = None) -> Value[D.T_value]:
        """
        Heuristic to be used by search algorithms, depending on the objective and constraints.

        Args:
            s (D.T_state): Actual state
            objective (str, optional): Objective function. Defaults to None.

        Returns:
            Value[D.T_value]: Heuristic value of the state.
        """

        # Current position
        pos = s.trajectory.iloc[-1]
        pos_alt = pos["alt"]
        pos_phase = self.network.nodes[s.id]["phase"]

        # Parameters
        lat_to, lon_to, alt_to = self.lat2, self.lon2, self.alt2
        lat_start, lon_start, alt_start = self.lat1, self.lon1, self.alt1

        cas = None
        mach = None

        # Determine current CAS/MACH
        if pos_phase == "climb":
            if pos_alt < 10_000:
                cas = min(self.climb_profile["below_10k_ft"], 250) * kts
            elif pos_alt < self.alt_crossover:
                cas = self.climb_profile["from_10k_to_crossover"] * kts
            else:
                mach = self.climb_profile["above_crossover"]
        elif pos_phase == "cruise":
            mach = self.cruise_profile["above_crossover"]
        elif pos_phase == "descent":
            if pos_alt < 10_000:
                cas = self.descent_profile["below_10k_ft"] * kts
            elif pos_alt < self.alt_crossover:
                cas = self.descent_profile["from_10k_to_crossover"] * kts
            else:
                mach = self.descent_profile["above_crossover"]

        if heuristic_name is None:
            heuristic_name = self.heuristic_name

        # Compute distance in meters
        distance_to_goal = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"] * ft),  # alt ft -> meters
            LatLon(lat_to, lon_to, height=alt_to * ft),  # alt ft -> meters
        )
        distance_to_start = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"] * ft),  # alt ft -> meters
            LatLon(lat_start, lon_start, height=alt_start * ft),  # alt ft -> meters
        )

        if heuristic_name == "distance":
            cost = distance_to_goal

        elif heuristic_name == "fuel":
            # bearing of the plane
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], lat_to, lon_to)

            # weather computations & A/C speed modification
            we, wn = 0, 0
            temp = 273.15
            if self.weather_interpolator:
                # wind computations
                wind_ms = self.weather_interpolator.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=pos["alt"], t=pos["ts"]
                )
                we, wn = wind_ms[2][0], wind_ms[2][1]

                # temperature computations
                temp = self.weather_interpolator.interpol_field(
                    [pos["ts"], pos["alt"], pos["lat"], pos["lon"]], field="T"
                )

                # check for NaN values
                if math.isnan(temp):
                    print("NaN values in temp")

            wspd = sqrt(wn * wn + we * we)

            if mach is not None:
                tas = mach2tas(mach, alt_to * ft)  # alt ft -> meters
            else:
                tas = cas2tas(cas, alt_to * ft)  # alt ft -> meters

            gs = compute_gspeed(
                tas=tas,
                true_course=radians(bearing_degrees),
                wind_speed=wspd,
                wind_direction=3 * math.pi / 2 - atan2(wn, we),
            )

            # override temp computation
            values_current = {
                "mass": pos["mass"],
                "alt": pos["alt"],
                "speed": tas / kts,
                "temp": temp,
            }

            # compute "time to arrival"
            dt = distance_to_goal / gs

            if distance_to_goal == 0:
                return Value(cost=0)

            if self.perf_model_name == "PS":
                cost = self.perf_model.compute_fuel_consumption(
                    values_current,
                    delta_time=dt,
                    path_angle=math.degrees(
                        (alt_to - pos["alt"]) * ft / (distance_to_goal)
                    )
                    # approximation for small angles: tan(alpha) ~ alpha
                )
            else:
                cost = self.perf_model.compute_fuel_consumption(
                    values_current,
                    delta_time=dt,
                )

        elif heuristic_name == "time":
            we, wn = 0, 0
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], self.lat2, self.lon2)

            if self.weather_interpolator:
                wind_ms = self.weather_interpolator.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=pos["alt"], t=pos["ts"]
                )

                we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300
            wspd = sqrt(wn * wn + we * we)

            if mach is not None:
                tas = mach2tas(mach, alt_to * ft)  # alt ft -> meters
            else:
                tas = cas2tas(cas, alt_to * ft)  # alt ft -> meters

            gs = compute_gspeed(
                tas=tas,
                true_course=radians(bearing_degrees),
                wind_speed=wspd,
                wind_direction=3 * math.pi / 2 - atan2(wn, we),
            )

            cost = distance_to_goal / gs

        elif heuristic_name == "lazy_fuel":
            fuel_consummed = s.trajectory.iloc[0]["mass"] - pos["mass"]
            cost = (
                1.05 * distance_to_goal * (fuel_consummed / (distance_to_start + 1e-8))
            )

        elif heuristic_name == "lazy_time":
            cost = (
                1.5
                * distance_to_goal
                * (
                    (pos["ts"] - s.trajectory.iloc[0]["ts"])
                    / (distance_to_start + 1e-8)
                )
            )
        else:
            cost = 0

        return Value(cost=cost)

    def set_network(
        self,
        p0: LatLon,
        p1: LatLon,

        climb_profile: Dict[str, float],
        cruise_profile: Dict[str, float],
        descent_profile: Dict[str, float],
        rate_of_climb_descent: Dict[str, float],
        
        steps: Dict[str, int],
        n_branches: int,

        plane_heading: float,

        graph_width: str = "medium",
    ):
        """
        Creation of the airway graph.

        Args:
            p0 : Origin of the flight plan
            p1 : Destination of the flight plan
            climb_profile: (Dict[str, float]): Climb profile of the aircraft in Kts (and Mach where applicable)
            cruise_profile: (float): Cruise profile of the aircraft in Kts (and Mach where applicable)
            descent_profile: (Dict[str, float]): Descent profile of the aircraft in Kts (and Mach where applicable)
            rate_of_climb_descent: (Dict[str, float]): Rate of climb and descent of the aircraft in ft/min
            steps: (Dict[str, float]): Number of steps for each phase of the flight plan
            n_branches: (int): Number of branches in the graph
            plane_heading: (float): Initial heading of the aircraft
            graph_width (str): Graph width of the graph. Defaults to medium

        Returns:
            A 3D matrix containing for each points its latitude, longitude, altitude between origin & destination.
        """

        # COORDINATES
        lon_start, lat_start = p0.lon, p0.lat
        lon_end, lat_end = p1.lon, p1.lat

        print(f"lat_start: {lat_start}, lon_start: {lon_start}; lat_end: {lat_end}, lon_end: {lon_end}")

        # CLIMB, DESCENT: rocd (in ft/min)
        rocd_climb = rate_of_climb_descent["climb"]
        rocd_descent = rate_of_climb_descent["descent"]

        # CLIMB: cas (in m/s) and mach
        cas_climb1 = min(climb_profile["below_10k_ft"], 250) * kts # min(cas, 250) * kts # kts => m/s, cas from 6k to 10k ft
        cas_climb2 = climb_profile["from_10k_to_crossover"] * kts # cas_climb2 * kts # kts => m/s, cas from 10k to crossover
        mach_climb = climb_profile["above_crossover"] # self.ac["cruise"]["mach"] # above crossover

        # DESCENT: cas (in m/s) and mach
        cas_descent1 = min(descent_profile["below_10k_ft"], 220) * kts # min(cas_descent1, 220) * kts # kts => m/s, cas from 3k to 6k ft
        cas_descent2 = min(descent_profile["below_10k_ft"], 250) * kts # min(cas_descent1, 250) * kts # kts => m/s, cas from 6k to 10k ft
        cas_descent3 = descent_profile["from_10k_to_crossover"] * kts # cas_descent3 * kts # kts => m/s, cas from 10k to crossover ft
        mach_descent = descent_profile["above_crossover"] # self.ac["cruise"]["mach"] # above crossover

        # CRUISE: mach
        mach_cruise = cruise_profile["above_crossover"] # mach_cruise
        assert mach_cruise < 1, "Mach number should be less than 1"

        # ALTITUDES
        alt_init = p0.height
        alt_toc = self.ac["cruise"]["height"] # from OpenAP
        alt_max = self.ac["limits"]["ceiling"] # from OpenAP
        alt_final = p1.height
        self.alt_crossover
        print(f"alt_crossover: {self.alt_crossover}")

        # HEADING, BEARING and DISTANCE
        total_distance = distance(lat_start, lon_start, lat_end, lon_end, h=int(alt_final - alt_init))
        half_distance = 3 * total_distance / 4

        # initialize an empty graph
        graph = nx.DiGraph()

        # first node is the origin
        graph.add_node(
            0, 
            node_id=0, 
            parent_id=-1, 
            branch_id=0, 
            lat=lat_start, 
            lon=lon_start, 
            height=alt_init, 
            heading=plane_heading,
            dist_destination=total_distance,
            dist_travelled=0,
            ts=0,
            phase="climb")

        # define the width of the graph
        if graph_width == "small":
            alpha = 10
        elif graph_width == "medium":
            alpha = 30
        elif graph_width == "large":
            alpha = 45
        elif graph_width == "xlarge":
            alpha = 60
        else:
            raise ValueError("Graph width not defined or incorrect.")
        
        angles = np.linspace(start=-alpha, stop=alpha, num=n_branches-1, endpoint=True, dtype=float)
        angles = np.sort(np.unique(np.append(angles, [0])))
        
        ########################################################################################################
        ######################### FLIGHT PHASES SETUP ##########################################################

        # CLIMB
        n_steps_climb = steps["n_steps_climb"]
        imposed_altitudes_climb = np.array([alt_init / ft, 10_000, self.alt_crossover, alt_toc / ft])
        possible_altitudes_climb = np.linspace(alt_init, alt_toc, num=n_steps_climb, endpoint=True) / ft
        possible_altitudes_climb = np.unique(np.sort(np.append(possible_altitudes_climb, imposed_altitudes_climb)))
        time_steps_climb = np.diff(possible_altitudes_climb) / rocd_climb * 60 # seconds

        # CRUISE
        n_steps_cruise = steps["n_steps_cruise"]
        n_steps_cruise_climb = steps["n_steps_cruise_climb"]
        possible_altitudes_cruise_climb = np.linspace(alt_toc, alt_max, num=n_steps_cruise_climb, endpoint=True)
        
        # DESCENT
        n_steps_descent = steps["n_steps_descent"]
        distance_start_descent = 150 * nm
        imposed_altitudes_descent_prep = np.array([self.alt_crossover, 10_000, alt_final / ft])
        
        ######################### END OF FLIGHT PHASES SETUP ###################################################
        ########################################################################################################

        ########################################################################################################
        ######################### BEGIN OF ALIGNMENT PHASE #####################################################

        # TODO: perform alignment
        # The goal is to align the heading of the aircraft with the destination in case that the initial heading 
        # is not the same as the destination. This will be done by adding nodes to the graph that will allow the
        # aircraft to align its heading with the destination. The aircraft will then follow the path to the destination.
        # The alignment phase will be done in the following way:
                # 1. Compute the angle between the initial heading and the destination
                # 2. Compute the distance to travel to align the heading with the destination
                # 3. Compute the number of steps to align the heading
                # 4. Compute the new heading after each step
                # 5. Compute the new position after each step
                # 6. Add the new node to the graph
                # 7. Add the new edge to the graph

        ######################### END OF ALIGNMENT PHASE #######################################################
        ########################################################################################################


        ########################################################################################################
        ######################### BEGIN OF FLIGHT PHASE ########################################################

        branches_ids = {"climb": [], "cruise": [], "cruise_correction": [], "descent": []}
        for branch_id in range(n_branches):
            parent_id = 0
            angle = angles[branch_id]
            distance_halfway = half_distance / math.cos(math.radians(angle))
            
            parent = graph.nodes[parent_id]
            parent_height = parent["height"]

            ###### CLIMB PHASE ######
            children_climb = []
            for index_climb, time_step_climb in enumerate(time_steps_climb):
                parent = graph.nodes[parent_id]
                parent_height = parent["height"]
                plane_heading_branch = aero_bearing(lat1=parent["lat"], lon1=parent["lon"], lat2=lat_end, lon2=lon_end) + angle

                # get the right speed according to the altitude
                if parent_height / ft < 10_000:
                    cas_climb = cas_climb1
                elif parent_height / ft < self.alt_crossover:
                    cas_climb = cas_climb2
                else:
                    cas_climb = mach2cas(mach_climb, parent_height)

                height = possible_altitudes_climb[index_climb+1] * ft
                dt = time_step_climb
                dx = cas_climb * dt

                # compute new position
                lat, lon = latlon(parent["lat"], parent["lon"], d=dx, brg=plane_heading_branch)

                # add the new node
                graph.add_node(
                    graph.number_of_nodes(),
                    node_id=graph.number_of_nodes(),
                    parent_id=parent_id,
                    branch_id=branch_id,
                    lat=lat,
                    lon=lon,
                    height=height,
                    heading=plane_heading_branch,
                    dist_destination=distance(lat, lon, lat_end, lon_end),
                    dist_travelled=parent["dist_travelled"] + dx,
                    ts=parent["ts"] + dt,
                    phase="climb"
                )

                # add the edge
                graph.add_edge(parent_id, graph.number_of_nodes()-1)

                parent_id = graph.number_of_nodes()-1

                children_climb.append(parent_id)
                
            branches_ids["climb"].append(children_climb)

            distance_climb_to_destination = graph.nodes[branches_ids["climb"][branch_id][-1]]["dist_destination"]
            distance_cruise = (distance_climb_to_destination - distance_start_descent)
            distance_step = distance_cruise / n_steps_cruise


            # PREPARING CRUISE, ALTITUDE CHANGES
            parent_id_after_climb = parent_id
            parent_ids_after_climb = []
            # FIRST CRUISE PHASE

            children_cruise = []
            for step_cruise_climb in range(n_steps_cruise_climb):
                children_cruise_climb = []
                plane_heading_branch = plane_heading + angle
                parent = graph.nodes[parent_id_after_climb]
                parent_height = parent["height"]
                target_altitude = possible_altitudes_cruise_climb[step_cruise_climb]
                plane_heading_branch = aero_bearing(lat1=parent["lat"], lon1=parent["lon"], lat2=lat_end, lon2=lon_end) + angle

                # Allows for a step climb during cruise
                if parent_height != target_altitude: 
                    cas_climb = mach2cas(mach_climb, parent_height)
                    dz_cruise_climb = (target_altitude - parent_height) / ft
                    dt_cruise_climb = dz_cruise_climb / rocd_climb * 60
                    dx_cruise_climb = cas_climb * dt_cruise_climb

                    # compute new position
                    lat, lon = latlon(parent["lat"], parent["lon"], d=dx_cruise_climb, brg=plane_heading_branch)

                    # add the new node
                    graph.add_node(
                        graph.number_of_nodes(),
                        node_id=graph.number_of_nodes(),
                        parent_id=parent_id,
                        branch_id=branch_id,
                        lat=lat,
                        lon=lon,
                        height=target_altitude,
                        heading=plane_heading_branch,
                        dist_destination=distance(lat, lon, lat_end, lon_end),
                        dist_travelled=parent["dist_travelled"] + dx_cruise_climb,
                        ts=parent["ts"] + dt_cruise_climb,
                        phase="cruise"
                    )

                    parent_id = graph.number_of_nodes()-1
                    children_cruise_climb.append(parent_id)

                for step_cruise in range(n_steps_cruise):
                    parent = graph.nodes[parent_id]
                    parent_distance_travelled = parent["dist_travelled"]
                    plane_heading_branch = aero_bearing(lat1=parent["lat"], lon1=parent["lon"], lat2=lat_end, lon2=lon_end) + angle

                    if parent_distance_travelled > distance_halfway:
                        plane_heading_branch = aero_bearing(parent["lat"], parent["lon"], lat_end, lon_end)

                    dx = distance_step
                    dt = dx / mach2cas(mach_cruise, height)

                    # compute new position
                    lat, lon = latlon(parent["lat"], parent["lon"], d=dx, brg=plane_heading_branch)

                    # add the new node
                    graph.add_node(
                        graph.number_of_nodes(),
                        node_id=graph.number_of_nodes(),
                        parent_id=parent_id,
                        branch_id=branch_id,
                        lat=lat,
                        lon=lon,
                        height=target_altitude,
                        heading=plane_heading_branch,
                        dist_destination=distance(lat, lon, lat_end, lon_end),
                        dist_travelled=parent["dist_travelled"] + dx,
                        ts=parent["ts"] + dt,
                        phase="cruise"
                    )

                    graph.add_edge(parent_id, graph.number_of_nodes()-1)

                    parent_id = graph.number_of_nodes()-1

                    children_cruise_climb.append(parent_id)
                
                parent_ids_after_climb.append(parent_id)
                children_cruise.append(children_cruise_climb)
            branches_ids["cruise"].append(children_cruise)

            
            children_cruise_correction = []
            for parent_group in branches_ids["cruise"][branch_id]:
                children_cruise_climb_correction = []
                parent_id_after_first_cruise = parent_group[-1]

                distance_after_cruise = graph.nodes[parent_id_after_first_cruise]["dist_destination"]
                imposed_descent_prep = np.unique(
                    np.sort(
                        np.concatenate(
                            (
                                [graph.nodes[parent_id_after_first_cruise]["height"]/ft], 
                                imposed_altitudes_descent_prep)
                            )
                        ))
                imposed_altitude_descent_diff = np.diff(imposed_descent_prep)
                cas_descent_profile = [mach2cas(mach_descent, graph.nodes[parent_id_after_first_cruise]["height"]), cas_descent3, cas_descent2, cas_descent1]
                imposed_times_step_descent = imposed_altitude_descent_diff / rocd_descent * 60

                # compute horizontal distance
                dx_total = 0
                for i, time_step_descent in enumerate(imposed_times_step_descent):
                    dx = cas_descent_profile[i] * time_step_descent
                    dx_total += dx

                delta_distance_cruise = distance_after_cruise - dx_total
                # print(f"To destination: {distance_after_cruise/nm}; In descent: {dx_total/nm}; To travel: {delta_distance_cruise/nm}")
                if delta_distance_cruise < 0:
                    raise ValueError("With the current ROCD and DESCENT speed profile, the plane cannot reach the destination altitude.")
                
                distance_step = delta_distance_cruise / 5


                parent_height = graph.nodes[parent_id_after_first_cruise]["height"]
                parent_id = parent_id_after_first_cruise
                dx_counter = 0
                for step_cruise in range(5):
                    parent = graph.nodes[parent_id]
                    parent_height = parent["height"]
                    parent_distance_travelled = parent["dist_travelled"]

                    plane_heading_branch = aero_bearing(parent["lat"], parent["lon"], lat_end, lon_end)
    
                    dx = distance_step
                    dt = dx / mach2cas(mach_cruise, height)
                    dx_counter += dx

                    # compute new position
                    lat, lon = latlon(parent["lat"], parent["lon"], d=dx, brg=plane_heading_branch)

                    # add the new node
                    graph.add_node(
                        graph.number_of_nodes(),
                        node_id=graph.number_of_nodes(),
                        parent_id=parent_id,
                        branch_id=branch_id,
                        lat=lat,
                        lon=lon,
                        height=parent_height,
                        heading=plane_heading_branch,
                        dist_destination=distance(lat, lon, lat_end, lon_end),
                        dist_travelled=parent["dist_travelled"] + dx,
                        ts=parent["ts"] + dt,
                        phase="cruise"
                    )

                    graph.add_edge(parent_id, graph.number_of_nodes()-1)

                    parent_id = graph.number_of_nodes()-1

                    children_cruise_climb_correction.append(parent_id)

                children_cruise_correction.append(children_cruise_climb_correction)
            branches_ids["cruise_correction"].append(children_cruise_correction)
            
            # DESCENT PHASE
            dx_counter = 0
            children_descent = []
            for parent_group in branches_ids["cruise_correction"][branch_id]:
                children_descent_group = []
                parent_id_after_cruise_correction = parent_group[-1]
                parent_height = graph.nodes[parent_id_after_cruise_correction]["height"]
                parent_id = parent_id_after_cruise_correction

                imposed_altitudes_descent = np.concatenate(([parent_height/ft], imposed_altitudes_descent_prep))
                possible_altitudes_descent = np.linspace(alt_final, parent_height, num=n_steps_descent, endpoint=True) / ft
                possible_altitudes_descent = np.unique(np.sort(np.append(possible_altitudes_descent, imposed_altitudes_descent)))[::-1]
                time_steps_descent = -np.diff(possible_altitudes_descent) / rocd_descent * 60 # seconds

                for index_descent, time_step_descent in enumerate(time_steps_descent):
                    parent = graph.nodes[parent_id]
                    parent_height = parent["height"]
                    plane_heading_branch = aero_bearing(parent["lat"], parent["lon"], lat_end, lon_end)

                    # get the right speed according to the altitude
                    if parent_height / ft < 10_000:
                        cas_descent = cas_descent2
                    elif parent_height / ft <= self.alt_crossover:
                        cas_descent = cas_descent3
                    else:
                        cas_descent = mach2cas(mach_descent, parent_height)

                    height = possible_altitudes_descent[index_descent+1] * ft
                    dt = time_step_descent
                    dx = cas_descent * dt
                    dx_counter += dx

                    # compute new position
                    lat, lon = latlon(parent["lat"], parent["lon"], d=dx, brg=plane_heading_branch)

                    # add the new node
                    graph.add_node(
                        graph.number_of_nodes(),
                        node_id=graph.number_of_nodes(),
                        parent_id=parent_id,
                        branch_id=branch_id,
                        lat=lat,
                        lon=lon,
                        height=height,
                        heading=plane_heading_branch,
                        dist_destination=distance(lat, lon, lat_end, lon_end),
                        dist_travelled=parent["dist_travelled"] + dx,
                        ts=parent["ts"] + dt,
                        phase="descent"
                    )

                    # add the edge
                    graph.add_edge(parent_id, graph.number_of_nodes()-1)

                    parent_id = graph.number_of_nodes()-1

                    children_descent_group.append(parent_id)

                children_descent.append(children_descent_group)
            # print(f"To destination: {graph.nodes[parent_id]['dist_destination']/nm}")
            branches_ids["descent"].append(children_descent)
        self.branches_ids = branches_ids

        ########################################################################################################
        ######################### START OF NODE CONNECTION #####################################################

        for branch_id in range(n_branches):
            ### CLIMB PHASE ###
            # connect to branch on the left
            if branch_id > 0:
                for parent_id, child_id in zip(branches_ids["climb"][branch_id][:-1], branches_ids["climb"][branch_id-1]):
                    graph.add_edge(parent_id, child_id+1)

            # connect to branch on the right
            if branch_id+1 < n_branches:
                for parent_id, child_id in zip(branches_ids["climb"][branch_id][:-1], branches_ids["climb"][branch_id+1]):
                    # print(f"{parent_id} -> {child_id+1}")
                    graph.add_edge(parent_id, child_id+1)

            parent_climb = branches_ids["climb"][branch_id][-1] # last climb node from the branch
            for altitude_index in range(n_steps_cruise_climb-1):
                child_cruise = branches_ids["cruise"][branch_id][altitude_index+1][0] # first cruise node from the branch at altitude above
                graph.add_edge(parent_climb, child_cruise)

            ### CRUISE + CORRECTION PHASES ###
            for altitude_index in range(n_steps_cruise_climb):
                current_altitude_nodes = branches_ids["cruise"][branch_id][altitude_index]
                
                ### CRUISE PHASE ###
                # connect to altitude on bottom
                if altitude_index > 0:
                    bottom_altitude_nodes = branches_ids["cruise"][branch_id][altitude_index-1]
                    for parent_id, child_id in zip(current_altitude_nodes, bottom_altitude_nodes):
                        if altitude_index - 1 == 0:
                            # print(f"{parent_id} -> {child_id}")
                            graph.add_edge(parent_id, child_id)
                        else:
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id+1)
                    
                    # connect to branch on the left
                    if branch_id > 0:
                        for parent_id, child_id in zip(current_altitude_nodes, bottom_altitude_nodes):
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id+1)

                    # connect to branch on the right
                    if branch_id+1 < n_branches:
                        for parent_id, child_id in zip(current_altitude_nodes, bottom_altitude_nodes):
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id+1)

                # connect to altitude on top
                if altitude_index+1 < n_steps_cruise_climb:
                    top_altitude_nodes = branches_ids["cruise"][branch_id][altitude_index+1] if altitude_index+1 < n_steps_cruise_climb else []
                    for parent_id, child_id in zip(current_altitude_nodes, top_altitude_nodes):
                        if child_id+1 in top_altitude_nodes:
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id+1)
                    
                    # connect to branch on the left
                    if branch_id > 0:
                        for parent_id, child_id in zip(current_altitude_nodes, top_altitude_nodes):
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id+1)

                    # connect to branch on the right
                    if branch_id+1 < n_branches:
                        for parent_id, child_id in zip(current_altitude_nodes, top_altitude_nodes):
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id+1)

                # connect to branch on the left
                if branch_id > 0:
                    for parent_id, child_id in zip(branches_ids["cruise"][branch_id][altitude_index], branches_ids["cruise"][branch_id-1][altitude_index]):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id+1)

                # connect to branch on the right
                if branch_id+1 < n_branches:
                    for parent_id, child_id in zip(branches_ids["cruise"][branch_id][altitude_index], branches_ids["cruise"][branch_id+1][altitude_index]):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id+1)

                ## CRUISE CORRECTION PHASE ###
                # connect to altitude on bottom
                if altitude_index > 0:
                    for parent_id, child_id in zip(branches_ids["cruise_correction"][branch_id][altitude_index], branches_ids["cruise_correction"][branch_id][altitude_index-1][:-1]):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id+1)

                # connect to altitude on top
                if altitude_index+1 < n_steps_cruise_climb:
                    for parent_id, child_id in zip(branches_ids["cruise_correction"][branch_id][altitude_index], branches_ids["cruise_correction"][branch_id][altitude_index+1][:-1]):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id+1)

                # connect to branch on the left
                if branch_id > 0:
                    for parent_id, child_id in zip(branches_ids["cruise_correction"][branch_id][altitude_index], branches_ids["cruise_correction"][branch_id-1][altitude_index]):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id+1)

                # connect to branch on the right
                if branch_id+1 < n_branches:
                    for parent_id, child_id in zip(branches_ids["cruise_correction"][branch_id][altitude_index], branches_ids["cruise_correction"][branch_id+1][altitude_index]):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id+1)

                ### DESCENT PHASE ###
                # connect to to branch on the left
                if branch_id > 0:
                    for parent_id, child_id in zip(branches_ids["descent"][branch_id][altitude_index], branches_ids["descent"][branch_id-1][altitude_index][:-1]):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id+1)

                # connect to branch on the right
                if branch_id+1 < n_branches:
                    for parent_id, child_id in zip(branches_ids["descent"][branch_id][altitude_index], branches_ids["descent"][branch_id+1][altitude_index][:-1]):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id+1)

        ######################### END OF NODE CONNECTION #######################################################
        ########################################################################################################

        return graph

    def get_network(self):
        return self.network

    def flying(
        self, 
        from_: pd.DataFrame, 
        to_: Tuple[float, float, int],
        phase: str
    ) -> pd.DataFrame:
        """Compute the trajectory of a flying object from a given point to a given point

        Args:
            from_ (pd.DataFrame): the trajectory of the object so far
            to_ (Tuple[float, float]): the destination of the object
            phase (str): the phase of the flight
        Returns:
            pd.DataFrame: the final trajectory of the object
        """
        pos = from_.to_dict("records")[0]
        pos_alt = pos["alt"]

        lat_to, lon_to, alt_to = to_[0], to_[1], to_[2]
        dist_ = distance(
            pos["lat"], pos["lon"], lat_to, lon_to, h=(alt_to - pos["alt"]) * ft
        )

        cas = None
        mach = None

        # determine current CAS/MACH
        if phase == "climb":
            if pos_alt < 10_000:
                cas = min(self.climb_profile["below_10k_ft"], 250) * kts
            elif pos_alt < self.alt_crossover:
                cas = self.climb_profile["from_10k_to_crossover"] * kts
            else:
                mach = self.climb_profile["above_crossover"]
        elif phase == "cruise":
            mach = self.cruise_profile["above_crossover"]
        elif phase == "descent":
            if pos_alt < 6_000:
                cas = min(self.descent_profile["below_10k_ft"], 220) * kts
            elif pos_alt < 10_000:
                cas = min(self.descent_profile["below_10k_ft"], 250) * kts
            elif pos_alt < self.alt_crossover:
                cas = self.descent_profile["from_10k_to_crossover"] * kts
            else:
                mach = self.descent_profile["above_crossover"]

        data = []
        epsilon = 100
        dt = 600
        dist = dist_
        loop = 0

        while dist > epsilon:
            # bearing of the plane
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], lat_to, lon_to)

            # wind computations & A/C speed modification
            we, wn = 0, 0
            temp = 273.15
            if self.weather_interpolator:
                time = pos["ts"] % (3_600 * 24)

                # wind computations
                wind_ms = self.weather_interpolator.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=alt_to, t=time
                )
                we, wn = wind_ms[2][0], wind_ms[2][1]

                # temperature computations
                temp = self.weather_interpolator.interpol_field(
                    [pos["ts"], pos["alt"], pos["lat"], pos["lon"]], field="T"
                )

            wspd = sqrt(wn * wn + we * we)

            if mach is not None:
                tas = mach2tas(mach, alt_to * ft)  # alt ft -> meters
            else:
                tas = cas2tas(cas, alt_to * ft)  # alt ft -> meters

            gs = compute_gspeed(
                tas=tas,
                true_course=radians(bearing_degrees),
                wind_speed=wspd,
                wind_direction=3 * math.pi / 2 - atan2(wn, we),
            )

            if gs * dt > dist:
                # Last step. make sure we go to destination.
                dt = dist / gs
                ll = lat_to, lon_to
            else:
                ll = latlon(
                    pos["lat"],
                    pos["lon"],
                    d=gs * dt,
                    brg=bearing_degrees,
                    h=int(alt_to * ft),
                )

            values_current = {
                "mass": pos["mass"],
                "alt": pos["alt"],
                "speed": tas / kts,
                "temp": temp,
            }

            pos["fuel"] = self.perf_model.compute_fuel_consumption(
                values_current,
                delta_time=dt,
                path_angle=math.degrees((alt_to - pos["alt"]) * ft / (gs * dt))
                # approximation for small angles: tan(alpha) ~ alpha
            )

            mass = pos["mass"] - pos["fuel"]

            # get new weather interpolators
            if pos["ts"] + dt >= (3_600.0 * 24.0):
                if self.weather_date:
                    if self.weather_date == self.initial_date:
                        self.weather_date = self.weather_date.next_day()
                        self.weather_interpolator = self.get_weather_interpolator()
            else:
                if self.weather_date != self.initial_date:
                    self.weather_date = self.weather_date.previous_day()
                    self.weather_interpolator = self.get_weather_interpolator()

            if mach is not None:
                new_row = {
                    "ts": (pos["ts"] + dt),
                    "lat": ll[0],
                    "lon": ll[1],
                    "mass": mass,
                    "cas_mach": mach,
                    "fuel": pos["fuel"],
                    "alt": alt_to,  # to be modified
                }
            if cas is not None:
                new_row = {
                    "ts": (pos["ts"] + dt),
                    "lat": ll[0],
                    "lon": ll[1],
                    "mass": mass,
                    "cas_mach": cas / kts,
                    "fuel": pos["fuel"],
                    "alt": alt_to,  # to be modified
                }

            dist = distance(
                ll[0],
                ll[1],
                lat_to,
                lon_to,
                h=(pos["alt"] - alt_to) * ft,  # height difference in m
            )

            if dist < dist_:
                data.append(new_row)
                dist_ = dist
                pos = data[-1]

            else:
                dt = int(dt / 10)
                print("going in the wrong part.")
                assert dt > 0

            loop += 1

        return pd.DataFrame(data)

    def get_weather_interpolator(self) -> Optional[GenericWindInterpolator]:
        weather_interpolator = None

        if self.weather_date:
            w_dict = self.weather_date.to_dict()

            mat = get_weather_matrix(
                year=w_dict["year"],
                month=w_dict["month"],
                day=w_dict["day"],
                forecast=w_dict["forecast"],
                delete_npz_from_local=False,
                delete_grib_from_local=False,
            )

            # returns both wind and temperature interpolators
            weather_interpolator = GenericWindInterpolator(file_npz=mat)

        return weather_interpolator

    def custom_rollout(self, solver, max_steps=100, make_img=True):
        observation = self.reset()

        solver.reset()
        clear_output(wait=True)

        # loop until max_steps or goal is reached
        for i_step in range(1, max_steps + 1):

            # choose action according to solver
            action = solver.sample_action(observation)

            # get corresponding action
            outcome = self.step(action)
            observation = outcome.observation


            print("step ", i_step)
            print("policy = ", action[0], action[1])
            print("New node id = ", observation.id)
            print("Alt = ", observation.alt)
            print("Cas/Mach = ", observation.trajectory.iloc[-1]["cas_mach"])
            print(observation)

            # if make_img:
            #     # update image
            #     plt.clf()  # clear figure
            #     clear_output(wait=True)
            #     figure = self.render(observation)
            #     # plt.savefig(f'step_{i_step}')

            # final state reached?
            if self.is_terminal(observation):
                break
        if make_img:
            print("Final state reached")
            clear_output(wait=True)
            fig = plot_full(self, observation.trajectory)
            plt.savefig(f"full_plot")
            plt.show()
            self.observation = observation
            pass
        # goal reached?
        is_goal_reached = self.is_goal(observation)

        terminal_state_constraints = self._get_terminal_state_time_fuel(observation)
        if is_goal_reached:
            if self.constraints is not None:
                if self.constraints["time"] is not None:
                    if (
                        self.constraints["time"][1] # type: ignore
                        >= terminal_state_constraints["time"]
                    ):
                        if (
                            self.constraints["fuel"]
                            >= terminal_state_constraints["fuel"]
                        ):
                            print(f"Goal reached after {i_step} steps!")
                        else:
                            print(
                                f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                            )
                    else:
                        print(
                            f"Goal reached after {i_step} steps, but not in the good timelapse!"
                        )
                else:
                    if self.constraints["fuel"] >= terminal_state_constraints["fuel"]:
                        print(f"Goal reached after {i_step} steps!")
                    else:
                        print(
                            f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                        )
            else:
                if self.ac["limits"]["MFC"] >= terminal_state_constraints["fuel"]:
                    print(f"Goal reached after {i_step} steps!")
                else:
                    print(
                        f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                    )
        else:
            print(f"Goal not reached after {i_step} steps!")

        return terminal_state_constraints, self.constraints


def compute_gspeed(
    tas: float, true_course: float, wind_speed: float, wind_direction: float
):
    # Tas : speed in m/s
    # course : current bearing
    # wind speed, wind norm in m/s
    # wind_direction : (3pi/2-arctan(north_component/east_component)) in radian
    ws = wind_speed
    wd = wind_direction
    tc = true_course

    # calculate wind correction angle wca and ground speed gs
    swc = ws / tas * sin(wd - tc)
    if abs(swc) >= 1.0:
        # Wind is to strong
        gs = tas
        error = "Wind is too strong"
    else:
        wca = asin(swc)  # * 180.0 / pi)
        gs = tas * sqrt(1 - swc * swc) - ws * cos(wd - tc)

    if gs < 0:
        # Wind is to strong
        gs = tas
        error = "Wind is too strong"
    else:
        # Reset possible status message
        error = ""
    return gs

# TODO: modify the function
def fuel_optimisation(
    origin: Union[str, tuple],
    destination: Union[str, tuple],
    actype: str,
    constraints: Optional[Dict[str, float]],
    weather_date: WeatherDate,
    solver_factory: Callable[[], Solver],
    max_steps: int = 100,
    fuel_tol: float = 1e-3,
) -> float:
    """
    Function to optimise the fuel loaded in the plane, doing multiple fuel loops to approach an optimal

    Args:
        origin (Union[str, tuple]):
            ICAO code of the departure airport of th flight plan e.g LFPG for Paris-CDG, or a tuple (lat,lon)

        destination (Union[str, tuple]):
            ICAO code of the arrival airport of th flight plan e.g LFBO for Toulouse-Blagnac airport, or a tuple (lat,lon)

        actype (str):
            Aircraft type describe in openap datas (https://github.com/junzis/openap/tree/master/openap/data/aircraft)

        constraints (dict):
            Constraints that will be defined for the flight plan

        wind_interpolator (GenericWindInterpolator):
            Define the wind interpolator to use wind informations for the flight plan

        fuel_loaded (float):
            Fuel loaded in the plane for the flight

        solver_factory:
            Solver factory used in the fuel loop

        max_steps (int):
            max steps to use in the internal fuel loop

        fuel_tol (float):
            tolerance on fuel used to stop the optimization

    Returns:
        float:
            Return the quantity of fuel to be loaded in the plane for the flight
    """

    small_diff = False
    step = 0

    new_fuel = constraints["fuel"]
    while not small_diff:
        domain_factory = lambda: FlightPlanningDomain(
            origin=origin,
            destination=destination,
            actype=actype,
            constraints=constraints,
            weather_date=weather_date,
            objective="distance",
            heuristic_name="distance",
            fuel_loaded=new_fuel,
            starting_time=0.0,
        )

        fuel_prec = new_fuel
        new_fuel = simple_fuel_loop(
            solver_factory=solver_factory,
            domain_factory=domain_factory,
            max_steps=max_steps,
        )
        step += 1
        small_diff = (fuel_prec - new_fuel) <= fuel_tol

    return new_fuel


def simple_fuel_loop(solver_factory, domain_factory, max_steps: int = 100) -> float:
    domain = domain_factory()
    with solver_factory() as solver:
        domain.solve_with(solver, domain_factory)
        observation: State = domain.reset()
        solver.reset()

        # loop until max_steps or goal is reached
        for i_step in range(1, max_steps + 1):

            # choose action according to solver
            action = solver.sample_action(observation)

            # get corresponding action
            outcome = domain.step(action)
            observation = outcome.observation

            if domain.is_terminal(observation):
                break

        # Retrieve fuel for the flight
        fuel = domain._get_terminal_state_time_fuel(observation)["fuel"]

    return fuel
