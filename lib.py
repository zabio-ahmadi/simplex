import numpy as np
from time import *
import sys
import os


class simplex:

    def __init__(self, path):

        self.start_time = time()
        self.path = path
        self.eps = 1e-10
        self.iteration = 0
        self.DEBUG = False
        self.standard_form = None
        self.A = None
        self.B = None
        self.C = None
        self.type = None
        self.LP = []
        self.tableau = []

        base_path = 'examples/'
        self.path = os.path.join(base_path, path)

    def debug(self, val: True):
        self.DEBUG = val

    def read_problem_from_file(self):
        """
        this fonction read problem from txt file 
        it read data according to defined standards that defined by Mr. eggenberg
        """
        with open(self.path, 'r') as data:
            lines = data.readlines()

        self.type = lines[0].split(";")[0]
        self.LP = [line.split(";")[:-1] for line in lines]
        # remove type from the first line
        del self.LP[0][0]

    def to_standard_form(self):
        """
            This function takes in the linear problem and converts it to standard form.
            It iterates through the LP, and converts the constraints to the standard form using if-else statements.
            example: 
            1x + 2x <= 3 = [1,2,<=,3]
            5x + 2x >= 4 = [-5,-2,<=,-4]
            3x + 2.5x = 3 = [3, 2.5,<=,3]  and  [-3, -2.5,<=,-3]
        """
        standard = []
        for line in self.LP[1:]:

            if '>=' in line:
                standard.append(
                    [-1 * float(el) if el != '>=' else '<=' for el in line])
            elif '=' in line:
                standard.append(
                    [float(el) if el != '=' else '<=' for el in line])
                standard.append(
                    [-1 * float(el) if el != '=' else '<=' for el in line])
            else:
                standard.append(
                    [float(el) if el != '<=' else el for el in line])

        self.A = [line[:len(line) - 2] for line in standard]
        self.B = [line[len(line) - 1:] for line in standard]

        self.C = [-1 * float(x) if self.type.upper() == 'MAX' else float(x)
                  for x in self.LP[0]]
        tableau = np.hstack(([self.A, np.eye(len(self.A)), self.B]))

        tableau = np.asarray(
            np.vstack((tableau, (self.C + [0.0] * (len(self.A) + 1)))), dtype=float)
        self.standard_form = standard
        self.tableau = tableau

    def print_standard(self):

        print("\n------------------------------ standard form -----------------------------------")
        print("min ", end="")
        print("-1(" if self.type.upper() == 'MAX' else "(", end="")
        [print(el, end='\t')for el in self.C]
        print(")\n")

        for line in self.standard_form:
            for el in line:
                print(f'{el}', end="\t")
            print()

        print("---------------------------------------------------------------------------------\n")

    def print_tableau(self, tableau):
        np.set_printoptions(suppress=True, linewidth=150)
        np.savetxt(sys.stdout, tableau, fmt=f"%8.3f")
        print()

    def convert_time(self, time):
        time = time * 1000
        if time > 0 and time < 1000:
            return f"{time:.1f} ms"
        elif time >= 1000 and time < 60000:
            return f"{time/1000:.2f} s"
        elif time >= 60000:
            time = time // 1000
            minutes = int(time // 60)
            seconds = time % 60
            return f"{minutes} m {seconds} s"

    def pivot(self, mat, pivot_index):
        """
        This function is used to find the pivot point in matrix.

        The function first finds the last row of the matrix and removes the last element of that row. Then, 
        it finds the first negative index in the last row.

        Next, the function initializes an empty list called "ratios" to store the ratios of each element in the matrix. 
        It iterates through each row of the matrix,
        and finds the ratio of the last element of the row divided by the element at the pivot index. 

        The function then finds the minimum ratio in the ratios list and finds the index of that minimum ratio. 
        """

        last_row = mat[-1, :]
        # remove last element of last row
        last_row = last_row[: -1]

        #  find the first negative index
        pivot_index = next(
            (index for index, value in enumerate(last_row) if value < 0), -1)
        ratios = []
        # Iterate through each row of the matrix
        # [mat.shape[0] - 2] excludes last two row of the matrix
        for row in range(mat.shape[0] - 2):
            pivot_value = mat[row, pivot_index]
            if pivot_value >= self.eps:
                ratios.append(
                    mat[row, -1] / pivot_value if mat[row, -1] / pivot_value >= 0 else np.inf)
            else:
                ratios.append(np.inf)

        # find minimum ration
        min_ratio = min(ratios, default=np.inf)
        # return minimum ratio index
        min_ratio_index = ratios.index(min_ratio)

        # if not found
        if pivot_index == -1:
            return None, None
        else:
            return min_ratio_index, pivot_index

    def solveTableau(self, mat):
        """
        The function uses the pivot element to transform the tableau into a new, 
        equivalent tableau that is closer to the optimal solution.
        """
        pivot_column = 0

        # while the tableau is not optimal
        while True:
            # get pivot row and column
            pivot_row, pivot_column = self.pivot(mat, pivot_column)
            self.iteration += 1

            pivot_value = mat[pivot_row, pivot_column]

            if self.DEBUG:

                if pivot_row != None:
                    print(f"pivot({pivot_row:}, {pivot_column:})")
                    print(f'pivot value: {pivot_value:.3f}')

                print(f"tableau: {self.iteration}")
                self.print_tableau(mat)
                print()

            # if it is an optimal tableau
            if pivot_column == None and pivot_row == None:
                return mat  #
            else:
                # for the same row : divide each row element by its value
                mat[pivot_row] /= mat[pivot_row, pivot_column]

                # for the other rows : use the formula
                for i_row in range(mat.shape[0]):
                    if i_row != pivot_row:
                        mat[i_row] -= mat[pivot_row] * mat[i_row, pivot_column]

    def getTableauPhase1(self, mat):
        """
        this function creates the first phase tableau of simplex. 
        """
        mat_old = mat
        # If there is a negative value at the last column
        # multiply it by -1 change it to positive one
        if any(t < 0 for t in mat[:-1, -1]):
            shape = mat.shape
            for i in range(mat.shape[0] - 1):
                if mat[i, -1] < 0:
                    mat[i] *= -1

            md = mat[::, -1]

            # Add identity matrix
            z2 = np.eye(mat.shape[0] - 1)
            z2 = np.vstack((z2, np.zeros(mat.shape[0] - 1)))

            mat = np.hstack((mat, z2))

            abs_md = abs(md)

            mat = np.hstack((mat, abs_md[np.newaxis].T))

            mat = np.vstack((mat, np.zeros((1, mat.shape[1]))))

            mat_old = np.vstack((mat_old, np.zeros((1, mat_old.shape[1]))))

            # Get the number of columns in the original matrix
            n_cols = len(mat_old[0])

            # calculate the sum of each column and put it at bottom (fonction objective)
            for col in range(mat.shape[1] - len(mat_old)):
                mat[-1, col] = -1 * mat[:-2, col].sum()

            for col in range(n_cols, mat.shape[1] - 1):
                mat[-1, col] = mat[:-2, col].sum()

            mat[-1, -1] = -1 * mat[:-2, -1].sum()

            mat = self.solveTableau(mat)

            if self.DEBUG:
                print('Solved tableau')
                print('--------------------------------')
                print('Tableau Simplexe:')
                print('--------------------------------')
                self.print_tableau(mat)
                print()

            return mat[:shape[0], :shape[1]]

        else:
            return mat

    def printSolution(self):
        """
        this fonction print the solution according to examples given by Mr. Eggenberg
        """
        self.read_problem_from_file()
        self.to_standard_form()
        self.print_standard()

        phase1 = self.getTableauPhase1(self.tableau)
        if self.DEBUG:
            print('--------------------------------')
            print('Tableau Simplexe:')
            print('--------------------------------')

            self.print_tableau(self.tableau)
            print("--------------------------------")
            print("Solving auxiliary problem\nAUX tableau:")

            print("--------------------------------")

            self.print_tableau(phase1)

        solution = self.solveTableau(phase1)
        print('--------- SOLUTION ----------')
        print(f'Obj Value :\t {-1 * solution[-1,-1]:.3f}')

        exec_time = (time() - self.start_time)
        print(
            f'Execution Time:\t {self.convert_time(exec_time)}')
        print(f'Iteration:\t {self.iteration}')
        print("-----------------------------")
