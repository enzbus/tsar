"""
Copyright © Enzo Busseti 2019.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)


def min_key(d):
    return min(d, key=d.get)


def greedy_grid_search(function,
                       parameter_indexables, num_steps=1):
    """Evaluates function on combinations from
    the parameter_indexables list. Returns lowest
    value obtained from greedy grid search, starting
    from the first parameters in the indexables."""

    results = {}

    current_counter = [0 for param in parameter_indexables]

    def evaluate_function():

        if not tuple(current_counter) in results:
            params = [param[current_counter[i]]
                      for i, param in enumerate(parameter_indexables)]

            logger.debug('evaluating function at %s' % params)
            value = function(*params)
            logger.debug('function value = %f' % value)
            results[tuple(current_counter)] = value

    evaluate_function()

    def search_n_step(n):
        for i in range(len(current_counter)):
            if current_counter[i] < len(parameter_indexables[i]) - 1:
                current_counter[i] += 1
                evaluate_function()
                if n > 1:
                    search_n_step(n - 1)
                current_counter[i] -= 1

            if current_counter[i] > 0:
                current_counter[i] -= 1
                evaluate_function()
                if n > 1:
                    search_n_step(n - 1)
                current_counter[i] += 1

    while True:

        search_n_step(num_steps)

        old_current_counter = current_counter
        current_counter = list(min_key(results))

        if tuple(old_current_counter) == tuple(current_counter):
            logger.debug('optimal function value = %f' %
                         results[tuple(current_counter)])
            logger.debug('optimal function parameters = %s' %
                         str(tuple(current_counter)))
            return results[tuple(current_counter)], \
                [param[current_counter[i]]
                 for i, param in
                 enumerate(parameter_indexables)]
