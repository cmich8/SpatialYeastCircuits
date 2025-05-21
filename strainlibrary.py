import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import ndimage
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Callable
from matplotlib import cm, colors
import matplotlib.animation as animation


ALPHA = "alpha_factor"
IAA = "IAA"
BETA = "beta_estradiol"
GFP = "GFP"
VENUS = "Venus"
BAR1 = "BAR1"
GH3 = "GH3"

# Constants for strain types
ACTIVATION = "activation"
REPRESSION = "repression"


def create_strain_library():
        """
        Create a library of all strains from the paper with growth parameters.
        
        Returns:
            Dictionary mapping strain IDs to StrainParameters objects
        """
        # Growth parameters dictionary
        growth_params = {
            'beta->alpha': {'strain': 'beta->alpha','k': 10.156865998540457,'r': 1.3877019835236286,'A': 3.811240700492937e-08,'lag': 8.065932810614516,'doubling_time': 0.4994928225150461,'r_squared': 0.9978592780369944},
            'alpha->venus': {'strain': 'alpha->venus','k': 10.156881939374474,'r': 1.3876456633006105,'A': 3.814254538133409e-08,'lag': 8.06591253743179,'doubling_time': 0.4995130953756935,'r_squared': 0.9978586598580573},
            'alpha->alpha': {'strain': 'alpha->alpha','k': 10.030632161041991,'r': 2.0645209329326293,'A': 1.0705887173990125e-08,'lag': 7.994289828061679,'doubling_time': 0.33574238434838116,'r_squared': 0.9999048003122031},
            'alpha->IAA': {'strain': 'alpha->IAA','k': 10.040333303759686,'r': 1.577804152831291,'A': 6.43804253236298e-09,'lag': 7.973971461454354,'doubling_time': 0.4393112917823972,'r_squared': 0.9999163020148677},
            'beta->IAA': {'strain': 'beta->IAA',
                'k': 10.040307128578494,
                'r': 1.9629605025697323,
                'A': 4.000104513358863e-08,
                'lag': 8.156733057057686,
                'doubling_time': 0.35311315721968883,
                'r_squared': 0.999744136703164},
            'IAA->GFP': {'strain': 'IAA->GFP',
                'k': 10.252654230445089,
                'r': 0.42549511085479685,
                'A': 0.0002582042194400119,
                'lag': 7.805876037701137,
                'doubling_time': 1.6290367688772023,
                'r_squared': 0.9955579879121793},
            'IAA->IAA': {'strain': 'IAA->IAA',
                'k': 10.009222723768396,
                'r': 0.9071483893517285,
                'A': 1.338045537808413e-09,
                'lag': 7.970174930108886,
                'doubling_time': 0.7640945943312384,
                'r_squared': 0.9999395888033465},
            'IAA->alpha': {'strain': 'IAA->alpha',
                'k': 15.947503133990207,
                'r': 0.6144061444836784,
                'A': 2.082679648386623e-08,
                'lag': 10.04798809953061,
                'doubling_time': 1.1281579567249245,
                'r_squared': 0.9904120220822631}}
                    
        strains = {}
        strains['IAA-|GFP'] = StrainParameters(
            strain_id='IAA-|GFP',
            input_molecule=IAA,
            regulation_type=REPRESSION,
            output_molecule=GFP,
            k1=1.76e6, d1=3.13e2, k2=8.29e5, K=2.57e2, n=0.89,
            d2=4.41e5, k3=2.12e5, d3=7.46e4, b=8.68e2
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA-|2xGFP'] = StrainParameters(
            strain_id='IAA-|2xGFP',
            input_molecule=IAA,
            regulation_type=REPRESSION,
            output_molecule=GFP,
            k1=0.69, d1=4.16e2, k2=2.77e2, K=1.79, n=1.011,
            d2=11.40, k3=0.0011, d3=0.49, b=0.0049
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA->GFP'] = StrainParameters(
            strain_id='IAA->GFP',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=GFP,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=3.15e4, d3=1.66e6, b=1.46e4
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha-|GFP'] = StrainParameters(
            strain_id='alpha-|GFP',
            input_molecule=alpha,
            regulation_type=REPRESSION,
            output_molecule=GFP,
            k1=6.05, d1=59.54, k2=10.61, K=0.67, n=0.80,
            d2=0.82, k3=3.66e-4, d3=1.10, b=0.0032
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->Venus'] = StrainParameters(
            strain_id='alpha->Venus',
            input_molecule=alpha,
            regulation_type=ACTIVATION,
            output_molecule=Venus,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=1.62e4, d3=2.15e6, b=5.96e3
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->GFP'] = StrainParameters(
            strain_id='beta->GFP',
            input_molecule=beta,
            regulation_type=ACTIVATION,
            output_molecule=GFP,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=0.089, d3=0.16, b=2.155e-4
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta-|GFP'] = StrainParameters(
            strain_id='beta-|GFP',
            input_molecule=beta,
            regulation_type=REPRESSION,
            output_molecule=GFP,
            k1=0.89, d1=0.17, k2=174.32, K=6.65, n=2.18,
            d2=0.56, k3=1.5e-4, d3=0.55, b=3.77e-4
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->IAA'] = StrainParameters(
            strain_id='alpha->IAA',
            input_molecule=alpha,
            regulation_type=ACTIVATION,
            output_molecule=IAA,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=566.24, d3=0.575, b=55.83
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->BAR1'] = StrainParameters(
            strain_id='beta->BAR1',
            input_molecule=beta,
            regulation_type=ACTIVATION,
            output_molecule=BAR1,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=26.038, d3=1.92e-6, b=0.35
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->IAA'] = StrainParameters(
            strain_id='beta->IAA',
            input_molecule=beta,
            regulation_type=ACTIVATION,
            output_molecule=IAA,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=3.48e3, d3=0.16, b=0.21
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->alpha'] = StrainParameters(
            strain_id='beta->alpha',
            input_molecule=beta,
            regulation_type=ACTIVATION,
            output_molecule=alpha,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=121.6, d3=0.062, b=0.14
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA->alpha'] = StrainParameters(
            strain_id='IAA->alpha',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=alpha,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=2.285, d3=0.28, b=0.74
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta->GH3'] = StrainParameters(
            strain_id='beta->GH3',
            input_molecule=beta,
            regulation_type=ACTIVATION,
            output_molecule=GH3,
            k1=50.66, d1=25.86, k2=11.00, K=57.12, n=1.26,
            d2=110.43, k3=109.96, d3=36.71, b=1.6e-4
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA->BAR1'] = StrainParameters(
            strain_id='IAA->BAR1',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=BAR1,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=1.89, d3=1.83e-13, b=0.365
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->GH3'] = StrainParameters(
            strain_id='alpha->GH3',
            input_molecule=alpha,
            regulation_type=ACTIVATION,
            output_molecule=GH3,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=582.32, d3=368.67, b=1.17e-14
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha-|IAA'] = StrainParameters(
            strain_id='alpha-|IAA',
            input_molecule=alpha,
            regulation_type=REPRESSION,
            output_molecule=IAA,
            k1=6.05, d1=59.54, k2=10.61, K=0.67, n=0.80,
            d2=0.82, k3=4.09, d3=8.74e-11, b=47.78
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta-|IAA'] = StrainParameters(
            strain_id='beta-|IAA',
            input_molecule=beta,
            regulation_type=REPRESSION,
            output_molecule=IAA,
            k1=0.89, d1=0.17, k2=174.32, K=6.65, n=2.18,
            d2=0.56, k3=2.024e11, d3=4.29e10, b=1.85e9
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['beta-|alpha'] = StrainParameters(
            strain_id='beta-|alpha',
            input_molecule=beta,
            regulation_type=REPRESSION,
            output_molecule=alpha,
            k1=0.89, d1=0.17, k2=174.32, K=6.65, n=2.18,
            d2=0.56, k3=0.077, d3=1.77, b=0.18
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA-|alpha'] = StrainParameters(
            strain_id='IAA-|alpha',
            input_molecule=IAA,
            regulation_type=REPRESSION,
            output_molecule=alpha,
            k1=1.76e6, d1=3.13e2, k2=8.29e5, K=2.57e2, n=0.89,
            d2=4.41e5, k3=471.73, d3=0.41, b=1.34
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha->alpha'] = StrainParameters(
            strain_id='alpha->alpha',
            input_molecule=alpha,
            regulation_type=ACTIVATION,
            output_molecule=alpha,
            k1=1.06e3, d1=0.245, k2=3.47e5, K=7.08e5, n=1.04,
            d2=1.56e5, k3=419.52, d3=2.10e4, b=2.32e4
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA->IAA'] = StrainParameters(
            strain_id='IAA->IAA',
            input_molecule=IAA,
            regulation_type=ACTIVATION,
            output_molecule=IAA,
            k1=8.95e4, d1=0.082, k2=1.73e7, K=1.24e7, n=0.836,
            d2=1.88e7, k3=775.05, d3=0.84, b=780.90
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['alpha-|BAR1'] = StrainParameters(
            strain_id='alpha-|BAR1',
            input_molecule=alpha,
            regulation_type=REPRESSION,
            output_molecule=BAR1,
            k1=6.05, d1=59.54, k2=10.61, K=0.67, n=0.80,
            d2=0.82, k3=0.0014, d3=3.74e7, b=1.096e8
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        strains['IAA-|GH3'] = StrainParameters(
            strain_id='IAA-|GH3',
            input_molecule=IAA,
            regulation_type=REPRESSION,
            output_molecule=GH3,
            k1=1.76e6, d1=3.13e2, k2=8.29e5, K=2.57e2, n=0.89,
            d2=4.41e5, k3=225.51, d3=42.46, b=3.14e-6
            k=growth_params['IAA->IAA']['k'],
            r=growth_params['IAA->IAA']['r'],
            A=growth_params['IAA->IAA']['A'],
            lag=growth_params['IAA->IAA']['lag']
        )

        