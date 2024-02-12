import math

def t_distribution_probability(degrees_of_freedom, z):
    """
    Calculate the probability for the t-distribution.

    Args:
        degrees_of_freedom (int): The degrees of freedom for the t-distribution.
        z (float): The value of z.

    Returns:
        float: The probability for the t-distribution.
    """
    numerator = math.gamma((degrees_of_freedom + 1) / 2)
    denominator = math.sqrt(degrees_of_freedom * math.pi) * math.gamma(degrees_of_freedom / 2)
    probability = numerator / denominator * (1 + (z ** 2) / degrees_of_freedom) ** (-(degrees_of_freedom + 1) / 2)
    return probability

degrees_of_freedom = int(input("Enter the degrees of freedom: "))
z_values = []
for i in range(3):
    z = float(input(f"Enter the value of z ({i+1}/3): "))
    z_values.append(z)

for z in z_values:
    probability = t_distribution_probability(degrees_of_freedom, z)
    print(f"Probability for z = {z}: {probability}")