# coding: utf-8
import abc
import logging
import time
import random
import sys

from random import random as rand
from typing import List, Tuple
from tqdm import tqdm


class AbstractMatrix(abc.ABC):
    """ Класс-интерфейс, в котором заданы 'необходимые' для объекта-матрицы методы """

    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """ Should return (#rows, #columns) """
        raise NotImplementedError

    @abc.abstractmethod
    def sum(self, other: "AbstractMatrix") -> "AbstractMatrix":
        """ Should return the sum of two matrices """
        raise NotImplementedError

    @abc.abstractmethod
    def matmul(self, other: "AbstractMatrix") -> "AbstractMatrix":
        """ Should return the result of multiplication of two matrices """
        raise NotImplementedError

    @abc.abstractmethod
    def non_zero_values_ratio(self) -> float:
        """ Should return the ratio of non-zero values in the matrix """
        raise NotImplementedError

    @abc.abstractmethod
    def transpose(self) -> "AbstractMatrix":
        """ Should return the transposed version of this matrix """
        raise NotImplementedError


class SimpleMatrix(AbstractMatrix):
    """ A naive implementation of matrix as a object """

    def __init__(self, list_of_lists: List[List[float]]):
        self.matrix = list_of_lists

    def shape(self):
        return len(self.matrix), len(self.matrix[0])

    def sum(self, other: "SimpleMatrix") -> "SimpleMatrix":
        h, w = self.shape()
        assert other.shape() == (h, w), f"sizes unfit for sum: {other.shape()} vs {(h, w)}"

        sum_matrix = []

        for row, row_other in zip(self.matrix, other.matrix):
            sum_matrix.append([element + element_other for element, element_other in zip(row, row_other)])

        return SimpleMatrix(sum_matrix)

    def matmul(self, other: "SimpleMatrix") -> "SimpleMatrix":

        (h, w), (h_other, w_other) = self.shape(), other.shape()
        assert w == h_other, f"sizes unfit for matmul, {w} <> {h_other}"

        # setting up a zero-valued matrix of the proper size
        mul_matrix = [[0 for _ in range(w_other)] for _ in range(h)]

        for i in range(h):
            for j in range(w_other):
                for k in range(w):
                    mul_matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]

        return SimpleMatrix(mul_matrix)

    def non_zero_values_ratio(self) -> float:
        h, w = self.shape()
        total_cells = h * w
        nz_values = [cell for row in self.matrix for cell in row if cell != 0]
        return len(nz_values) / total_cells

    def transpose(self) -> "SimpleMatrix":
        raise NotImplementedError


class SparseMatrix(AbstractMatrix):
    """
        The smarter representation of sparse matrices fit for faster multiplication;
    """

    def __init__(self, list_of_lists: List[List[float]]):
        self.matrix = list_of_lists
        self.non_zero_col = []  # Номера ненулевых столбцов по строке
        self.non_zero_num = [
            0]  # Количество ненулевых стобцов в i-ой строке = self.non_zero_num[i+1]-self.non_zero_num[i]
        self.non_zero_el = []  # Соответсвующий ненулевой элемент
        num_of_non_zero = 0
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                if list_of_lists[i][j] != 0:
                    self.non_zero_col.append(j)
                    num_of_non_zero += 1
                    self.non_zero_el.append(list_of_lists[i][j])
            self.non_zero_num.append(num_of_non_zero)

    def shape(self):
        return len(self.matrix), len(self.matrix[0])

    def sum(self, other: "SparseMatrix") -> "SparseMatrix":
        h, w = self.shape()
        assert other.shape() == (h, w), f"sizes unfit for sum: {other.shape()} vs {(h, w)}"

        sum_matrix = [[0 for _ in range(w)] for _ in range(h)]

        for i in range(h):
            for j in range(self.non_zero_num[i], self.non_zero_num[i + 1]):
                sum_matrix[i][self.non_zero_col[j]] += self.non_zero_el[j]

        for i in range(h):
            for j in range(other.non_zero_num[i], other.non_zero_num[i + 1]):
                sum_matrix[i][other.non_zero_col[j]] += other.non_zero_el[j]

        return SparseMatrix(sum_matrix)

    def matmul(self, other: "SparseMatrix") -> "SparseMatrix":
        (h, w), (h_other, w_other) = self.shape(), other.shape()
        assert w == h_other, f"sizes unfit for matmul, {w} <> {h_other}"

        mul_matrix = [[0 for _ in range(w_other)] for _ in range(h)]

        other = other.transpose()  # Транспонируем вторую матрицу

        for i in range(h):
            for column in self.non_zero_col[self.non_zero_num[i]:self.non_zero_num[i + 1]]:
                for j in range(w_other):
                    if column in other.non_zero_col[other.non_zero_num[j]:other.non_zero_num[j + 1]]:
                        mul_matrix[i][j] += self.matrix[i][column] * other.matrix[j][column]

        return SparseMatrix(mul_matrix)

    def non_zero_values_ratio(self) -> float:
        h, w = self.shape()
        total_cells = h * w
        return len(self.non_zero_el) / total_cells

    def transpose(self) -> "SparseMatrix":
        h, w = self.shape()

        transposed_matrix = [[0 for _ in range(h)] for _ in range(w)]

        for i in range(h):
            for j in self.non_zero_col[self.non_zero_num[i]:self.non_zero_num[i + 1]]:
                transposed_matrix[j][i] = self.matrix[i][j]

        return SparseMatrix(transposed_matrix)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.info("Setting up the task...")

    # setting up large sparse matrices
    random.seed(261)
    m, n, l = 700, 300, 600
    zero_prob = 0.93

    matrix_a = [[0 if rand() < zero_prob else rand() for _ in range(n)] for _ in tqdm(range(m), "rows A")]
    matrix_a1 = SimpleMatrix(matrix_a)
    matrix_a2 = SparseMatrix(matrix_a)
    matrix_b = [[0 if rand() < zero_prob else rand() for _ in range(l)] for _ in tqdm(range(n), "rows B")]
    matrix_b1 = SimpleMatrix(matrix_b)
    matrix_b2 = SparseMatrix(matrix_b)
    
    # for SimpleMatrix
    logging.info(f"The sizes: A_{matrix_a1.shape()}, B_{matrix_b1.shape()}")
    logging.info(f"Sparsity: "
                 f"A ~ {matrix_a1.non_zero_values_ratio():.3f}, "
                 f"B ~ {matrix_b1.non_zero_values_ratio():.3f}")

    logging.info("Computing the sum...")
    start_sum = time.time()
    matrix_c1 = matrix_a1.sum(matrix_a1)
    end_sum = time.time()
    logging.info(f"Elapsed time for sum: {end_sum - start_sum:.4f} seconds")  # ~ 0.04 seconds

    logging.info("Computing the matmul...")
    start_mul = time.time()
    matrix_d1 = matrix_a1.matmul(matrix_b1)
    end_mul = time.time()
    logging.info(f"Elapsed time for matmul: {end_mul - start_mul:.4f} seconds")  # ~ 60 seconds

    logging.info("Done for SimpleMatrix.")

    # for SparseMatrix
    logging.info(f"The sizes: A_{matrix_a2.shape()}, B_{matrix_b2.shape()}")
    logging.info(f"Sparsity: "
                 f"A ~ {matrix_a2.non_zero_values_ratio():.3f}, "
                 f"B ~ {matrix_b2.non_zero_values_ratio():.3f}")

    logging.info("Computing the sum...")
    start_sum = time.time()
    matrix_c2 = matrix_a2.sum(matrix_a2)
    end_sum = time.time()
    logging.info(f"Elapsed time for sum: {end_sum - start_sum:.4f} seconds")  # ~ 0.04 seconds

    logging.info("Computing the matmul...")
    start_mul = time.time()
    matrix_d2 = matrix_a2.matmul(matrix_b2)
    end_mul = time.time()
    logging.info(f"Elapsed time for matmul: {end_mul - start_mul:.4f} seconds")  # ~ 60 seconds

    logging.info("Done for SparseMatrix.")
