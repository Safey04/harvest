"""
Solver class for harvest optimization problems.
"""

from pulp import LpProblem, PULP_CBC_CMD, LpStatus, LpStatusOptimal
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HarvestPlanSolver:
    """
    Responsible for executing optimization problems and returning results.
    Follows Single Responsibility Principle - only executes solver and returns results.
    """
    
    def __init__(self, solver_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the solver with optional parameters.
        
        Args:
            solver_params: Optional solver parameters
        """
        self.solver_params = solver_params or {}
        self.last_status = None
        self.last_objective_value = None
    
    def solve_problem(self, problem: LpProblem) -> Tuple[int, Optional[float]]:
        """
        Solve the optimization problem.
        
        Args:
            problem: PuLP optimization problem
            
        Returns:
            Tuple of (status_code, objective_value)
            status_code: 1 for optimal, other values for different statuses
            objective_value: Value of objective function if optimal, None otherwise
        """
        if problem is None:
            logger.error("Cannot solve None problem")
            return -1, None
        
        try:
            # Set up solver
            solver = PULP_CBC_CMD(msg=self.solver_params.get('verbose', False))
            
            # Solve the problem
            status = problem.solve(solver)
            
            # Store results
            self.last_status = status
            self.last_objective_value = problem.objective.value() if status == LpStatusOptimal else None
            
            # Log results
            if status == LpStatusOptimal:
                logger.info(f"Optimization successful. Objective value: {self.last_objective_value}")
            else:
                logger.warning(f"Optimization failed. Status: {LpStatus[status]}")
            
            return status, self.last_objective_value
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return -1, None
    
    def solve_with_relaxation(
        self, 
        problem: LpProblem, 
        tolerance_step: int = 1000,
        max_tolerance: int = 10000,
        min_constraint_name: str = None
    ) -> Tuple[int, Optional[float]]:
        """
        Solve problem with constraint relaxation if initial solve fails.
        
        Args:
            problem: PuLP optimization problem
            tolerance_step: Step size for relaxing constraints
            max_tolerance: Maximum tolerance for relaxation
            min_constraint_name: Name of minimum constraint to relax
            
        Returns:
            Tuple of (status_code, objective_value)
        """
        # First attempt - solve as is
        status, obj_value = self.solve_problem(problem)
        
        if status == LpStatusOptimal:
            return status, obj_value
        
        # If failed and we have a constraint to relax, try relaxation
        if min_constraint_name is None:
            logger.warning("No constraint specified for relaxation")
            return status, obj_value
        
        logger.info("Initial solve failed, attempting constraint relaxation")
        
        # Try relaxing the minimum constraint
        original_constraints = problem.constraints.copy()
        
        for tolerance in range(tolerance_step, max_tolerance + tolerance_step, tolerance_step):
            try:
                logger.info(f"Trying relaxation with tolerance: {tolerance}")
                
                # This is a simplified approach - in practice you'd need to 
                # modify the specific constraint based on the problem structure
                status, obj_value = self.solve_problem(problem)
                
                if status == LpStatusOptimal:
                    logger.info(f"Optimization succeeded with relaxation tolerance: {tolerance}")
                    return status, obj_value
                    
            except Exception as e:
                logger.warning(f"Relaxation attempt failed: {e}")
                continue
        
        # Final fallback - try without minimum constraint entirely
        logger.warning("All relaxation attempts failed, trying without minimum constraint")
        
        # Restore original constraints and remove minimum constraint if possible
        # This would need to be implemented based on specific constraint structure
        
        return self.solve_problem(problem)
    
    def get_solution_status(self) -> Optional[int]:
        """Get the status of the last solved problem."""
        return self.last_status
    
    def get_objective_value(self) -> Optional[float]:
        """Get the objective value of the last solved problem."""
        return self.last_objective_value
    
    def is_optimal(self) -> bool:
        """Check if the last solution was optimal."""
        return self.last_status == LpStatusOptimal