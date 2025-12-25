
{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w5mTdARoNYVb",
        "outputId": "98ffd901-0e33-4ca0-f1ec-be6a4c94d073"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Path to dataset files: /kaggle/input/fer2013\n"
          ]
        }
      ],
      "source": [
        "#dataset link https://www.kaggle.com/datasets/msambare/fer2013\n",
        "\n",
        "import kagglehub\n",
        "\n",
        "# Download latest version\n",
        "path = kagglehub.dataset_download(\"msambare/fer2013\")\n",
        "\n",
        "print(\"Path to dataset files:\", path)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "wd1UQBs7Nnfs",
        "collapsed": True

      },
      "outputs": [],
      "source": [
        "# STEP 1: Install TensorFlow (optional in Colab, already included)\n",
        "# !pip install tensorflow\n",
        "\n",
        "# STEP 2: Import Libraries\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense\n",
        "import matplotlib.pyplot as plt\n",
        "from os import walk  # Only for counting images\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "F7QW0ss7SEBx",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 839
        },
        "outputId": "6c43877e-eece-41ea-d23e-5b828a3d627a"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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ma9VV1cXL7zwQuzYsSPzZy6++OIoFAryLiEi0p3AT37yk+jp6ZE/eyo+//nPZ8YWL16c/r2kpCQWL14cR44ciaeeeqrfzy1YsCAaGhrk+wwePDhKS0vjmWeeif3797/rz6xcuTI6Ojrimmuuib1796Z/Bg0aFBdeeGH89Kc/PYVPhg8iksKHyAUXXBCXXXZZv39O9M5f5WzZsiUifvkl/k6TJk1K8be9/Tv/E9XW1kZLS8tJX+K1tbWZXyxve+utt6KpqSmGDh2qP9g79PX1xd/93d/F+PHjo6ysLOrr66OhoSFWrVqVfncfEfG1r30tqqqq4oILLojx48fHF7/4xXj++ef7vdZdd90Vr7/+eowaNSouuOCCuO2222wyy9La2hpf/vKX45//+Z+jvr4+5syZE/fee2+/bRro676b0047LcaOHdvvzyZMmBARcdLfAbhf40X8MtneeeedsWLFihg+fHj6NeGuXbvSz2zYsCEiIj71qU9FQ0NDv3+efPLJ2L1796l8NHwAkRQ+Qk68SxiIQYMGndKfF96jJ73efvvt8eUvfzlmz54d//Iv/xI/+clPYuXKlTFlypR+vyc/++yzY926dfHQQw/FRRddFI8++mhcdNFF/e5Mrrrqqti4cWN85zvfiaamprj77rtjypQpsWLFigFt27e+9a1YtWpV3HzzzXHo0KG46aabYsqUKbFt27aIyL4DOn78eOZr5j1up/IaX/rSl2L9+vXxjW98I8rLy+Ov/uqv4uyzz45XXnklIiLt3wceeOCku9KVK1fGj3/849zbivcXSeEjbPTo0RERsW7dupNi69atS/H3yrhx42LHjh3R3t5+SuseeeSRuOSSS+L73/9+XH311fHpT386Lrvssujo6DjpZysrK+N3f/d3Y9myZbF169aYP39+fP3rX4/e3t70MyNHjowbbrghfvSjH8WmTZti2LBh8fWvf33An2vq1Klxyy23xLPPPhvPPfdcbN++Pe67776IiBgyZEhExEnb+s67smL09fWddFezfv36iIhcf9k7bty4+PM///N48skn4/XXX48jR47Et771rRSLiGhsbDzprvSyyy6Liy++eMDviw8GksJH2IwZM6KxsTHuu+++OHz4cPrzFStWxJo1a2L+/Pnv6fsvWLAgCoVC/PVf//VJMXWXMWjQoJPiDz/88ElltPv27ev336WlpTF58uQoFApx9OjROH78+Em/2mlsbIympqZ++6PYktQDBw7EsWPH+v3Z1KlT47TTTkuvV1NTE/X19fHss8/2+7nvfve78rWz3HPPPenfC4VC3HPPPXHGGWfEpZdeesqv1dPT0y9ZRvwyCVRXV6ftnzNnTtTU1MTtt98eR48ePek19uzZc8rviw8WSlI/ws4444y48847Y9GiRfHJT34yrrnmmlSSOmbMmPizP/uz9/T9L7nkkvjc5z4X//AP/xAbNmyIuXPnRl9fXzz33HNxySWX9PtL1BNdfvnlsXTp0li0aFHMmjUrXnvttXjwwQdP+v36pz/96RgxYkR8/OMfj+HDh8eaNWvinnvuifnz50d1dXV0dHRES0tLfPazn41p06ZFVVVVPPXUU/HSSy+l/zOOKL4k9emnn47FixfHlVdeGRMmTIhjx47FAw88EIMGDYoFCxakn7v++uvjjjvuiOuvvz5mzJgRzz77bPo//FNRXl4eTzzxRCxcuDAuvPDCWLFiRSxfvjxuvvlm+5fK72b9+vVx6aWXxlVXXRWTJ0+O008/Pf793/892tra4uqrr46IXya1733ve/G5z30upk+fHldffXU0NDTE1q1bY/ny5fHxj3+8X6LCrx+SwkfcddddFxUVFXHHHXfE1772taisrIzPfOYzceedd/brUXivLFu2LM4999z4/ve/H1/96lejtrY2ZsyYEbNmzcpcc/PNN0d3d3f84Ac/iH/913+N6dOnx/Lly0+a8/Qnf/In8eCDD8bf/u3fRldXV7S0tMRNN90Ut9xyS0REVFRUxA033BBPPvlk/Nu//Vv09fXFWWedFd/97nfjC1/4wil/lmnTpsWcOXPisccei+3bt0dFRUVMmzYtVqxYETNnzkw/d+utt8aePXvikUceiR/+8Icxb968WLFiRTQ2Np7S+w0aNCieeOKJ+MIXvhBf/epXo7q6OpYsWRK33nrrKW97RMSoUaPimmuuif/8z/+MBx54IE4//fSYNGlS/PCHP+yX1K699tpoamqKO+64I+6+++44fPhwNDc3xyc+8YlYtGjRgN4bHxwlhffqbwMBvGeuu+66eOSRR6Krq+v93hR8yPB3CgCAhKQAAEhICgCAhL9TAAAk3CkAABKSAgAgKbpPYdKkSTKuHuk3fPhwuVZNxIwI+3Qo9fSwrVu3yrWHDh2S8bKysszYkSNH5Fr3m7l3do8W+74REVVVVTLumpdO7Ng91dfOmnX0NnU8S0tL5Vr3zIETnw/xbt5+lsK7GTFihFzrztO3R1S8GzfUzx1Pt09PPz37UnXn2bt1Hp9InYfu+ti0aZOMnzi19p3eObTvVF/7xEF970Z1V6v9GeGvAfV9986u9lONO+r7UB3LiIg1a9bY1+dOAQCQkBQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQFN2n4HoJhg0bNuC1ro7a1XC/8wlbJ3q3RzSeSPU4ROh6fldz7x7YrmrX3XZVVFTIuKuFzvPcX/e51ba5PgTVC1BMXJ2H7vkQlZWVMq72mapbj/B18Xn6FBx3/anXdtvV2toq4yc+M/ud1HOpI/I/xU1te1tbm1zrvjfUeVxTUyPXur4Sd6zVte2OVzG4UwAAJCQFAEBCUgAAJCQFAEBCUgAAJCQFAEBSdJ2bGyWrSigPHjwo17pyPlfauWPHjsyYK3tzY7kV97lcKaDaZ2678pacqrI3tzbPeGtVMhrhS07deajWu+12Zb7qPHXHOi9VxujeO0+ZojsPXfmkKll1pZluNL0qRY/QY7+rq6vlWvedo+JuvLsbs+6+V9Q14MrFi8GdAgAgISkAABKSAgAgISkAABKSAgAgISkAABKSAgAgKbpPwdVwq/HXavx0hK913r9/v4yremRXt+tqpVWfQ29vr1yrRmNH6H2at4bb1a6rXoTGxka51vUaqPVufLWrH3d9Cmqfuv6LPH0KbtR53j6GPH0K7+W2ufNUjZGeMGFCrtdub2+XcfW9kXefqN6PvXv3yrVqnHiE79VR2+YeQ1AM7hQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQkBQAAEnRfQquJj/PswG6u7tlvKOjQ8ZVL4GrCXY9FKru19Uyu/4Ltd1ureu/cP0AZ555ZmbM9SHkmRdfWVkp1+aNq3PNnYfuuR6utl1xNfcu7s5jxW23qrl3z+1w56l6b7ddY8aMkfGZM2fK+JYtWzJjqrcpwn9u9SwH973g+isc1ceQ59kZb+NOAQCQkBQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQFN2n4GqKe3p6MmOuPtw9L6Grq0vGVY13nudAuNd2Ndourmqh3ex/98yD1tZWGW9qasqMqT6DCN8DoWbou/4Kd664/aJe3x0PV1+u4nn7EFxccT0M7nOpfhlX9+5eW11fri9EnUcREePHj5fxGTNmZMa2bdsm17reKHWeut4n913qesI6OzszY+q6LhZ3CgCAhKQAAEhICgCAhKQAAEhICgCAhKQAAEiKLkl15WGqvNKVdx08eFDGXcmdKjV0JXVuRK4qFXSvrUr9IiKqqqoGFIuIGDdunIy3tLTIuCppdSWpecZXu332fo6vdlT5ZZ5y1rzxPOOrI/KVXbvrR8l7LrhrZMqUKZmx//3f/5Vr29raZFx9J7nzzH0uVd4foa9dN/a+GNwpAAASkgIAICEpAAASkgIAICEpAAASkgIAICEpAACSovsUHFUf68bUutHYrp5Z1VK7mt88Y4dd/berR1YmTZok46NGjZJxN1p7xIgRmTE3vtqNv1bHyx1LF3dUjbjrBchTc+9e23G17Wq/uLV5znE3Wt59bnUNHDlyZMBrI3y/jBojPXHiRLl27dq1Mn7gwIHMmNtn7nMPGTJExtXnXrNmjVxbDO4UAAAJSQEAkJAUAAAJSQEAkJAUAAAJSQEAkJAUAABJ0X0KO3bs0C8kegXq6+vl2r1798q4q4VW73348GG5Nk/dvNsu18egnongnoeg+gwi/DMRysrKMmOuD8HN2Ffy1vO7mnsVz/MsBsd9LvdsDdcb4tYrrt5f9Tm4feaOh5LnOQ8R/nMNHjw4MzZhwgS5dvjw4TK+Z8+ezJjbbvdsmtraWhlft27dgNcWgzsFAEBCUgAAJCQFAEBCUgAAJCQFAEBCUgAAJEXXFvb29sr4vn37MmOqNCzCl1+2t7fLuCrXcyWnboxtRUVFZsyVIVZVVcm4Gm89cuRIudaVnrmyUrXP3Ahp97nVPnelle6185SkurWu1FaNRM772m7Euyohdue4u3bVtr2XJamOK+10cfW9U1dXJ9eeeeaZMq5GVLvR1y6+ceNGGVfXdp7S5bdxpwAASEgKAICEpAAASEgKAICEpAAASEgKAICEpAAASIruU1A12o4bu+3GQLe2tsr41q1bM2Ou5l7Vf0foGm437th9LtXH4OraXX14V1fXgN/b7TM3sljVzbu69zyjzCN0n4Ora3dj1tUxcX0Gbrvd8VTnmntt17OienXcsXbnqdvnSt7R2uo7y/X5uN6q8ePHZ8ZUz1ZExNq1a2Xc9erk6ZcpBncKAICEpAAASEgKAICEpAAASEgKAICEpAAASEgKAICk6D4FV4etattdrbOr61XPNIiIGD16dGZs/fr1cq2r4VZ12K5G281NV7XS7jkPrg/B7bM8z79wddQq7mrq3bni4qq2Pe98frXtrj7cfW53LqkeClfP757robbNnYfuvdV695nzPicizzMohg0bNuD4m2++Kdc67vpS+5znKQAAfqVICgCAhKQAAEhICgCAhKQAAEhICgCApOiSVFfqdOjQocyYK1tzY2y3b98u46ocdtKkSXKtG+utSnFdyenBgwdlfO/evZkxt8/2798v425styrXc2WI7lxQ5ZnvdUlqnrHdLp5nrdunLq7OQ7fWjQRXx7Ojo0OubWxslPGhQ4dmxlwJcHV1tYy7slFVbu5eW5W5R0Q89thjmTH1XRjhS77d6Ho1OtuV2BeDOwUAQEJSAAAkJAUAQEJSAAAkJAUAQEJSAAAkJAUAQFJ0n4KrnVX1426tq7OurKyU8d27d2fG3HjelpYWGX/rrbcyY65XwNWHb9u2LTPmeiBUn0GE7/0YOXJkZqy8vFyudb0Gqgbc1Ye7en9X267OQ7fP3Hmq1rueFDdO2fXi7Nq1KzPm6uLdWG9VNz9x4kS5VvUhROhzxV2bBw4ckHF3fTU1NWXGXK+A64dR2+62y51nbnT2e407BQBAQlIAACQkBQBAQlIAACQkBQBAQlIAACQkBQBAUnSfQm9vr4yr+nE349vV5ar54RG65tjNg3f1yDNmzMiMufrwPP0Zqvciwu+zTZs2yfjLL7+cGXP1/BUVFTKuZtGPGzdOrnX9Ga623fVYKO45EZ2dnZmxV199Va5dtWqVjLteHfW5q6qq5Nrm5mYZV/v8zDPPlGvde6tzqaamRq6tr6+XcdfToq4R9XyKYt5b9TetWbNGrnXnWR6uj6cY3CkAABKSAgAgISkAABKSAgAgISkAABKSAgAgISkAAJKi+xRcPb/qJXBr3bz3PPXIrqbezcFXrz19+nS5ds+ePTKuej9cvbGqmS+G6/1QXM+K2jY3I989B8L1IaieFdd/4WrXu7q6MmNuf6rejQj/uVWvj9un7lxqb2/PjLlnA7h9pvaL6+MZPny4jLt95vqIFNdDoZ4j4da657C4/aJ6VvJc12/jTgEAkJAUAAAJSQEAkJAUAAAJSQEAkJAUAABJ0SWprrxLlQK6UbGuZM6NSx40aNCA39uVKapyPzeeWpWtRejx2K60zJWtuTJeVSbsSv3UsY7Q5ZPqWBUTHzFihIyrMdBun+7du1fG1ZhoN2LaXT8uvnPnzsyYK6t2Zbwq7q4fNyZ63759mTE1lj7Cj+V23xvqPHajyl1ZaWtra2assrJSrnXj/N01oI6Je+9icKcAAEhICgCAhKQAAEhICgCAhKQAAEhICgCAhKQAAEiK7lNw9fyqZtjVE+el6npdvb6rw1ajgXfs2CHXutHZqt5/y5Ytcq0baVxXVyfjap+513YjjdW4cjcm3W13fX29jKt+AXceutp01eegek6K4cYpt7W1ZcbcaHpXk6+Olzse5557royr/gvVwxARUV1dLeOuX0Zd+26t+14466yzMmPNzc1yretT6O7ulnF3vPPiTgEAkJAUAAAJSQEAkJAUAAAJSQEAkJAUAAAJSQEAkBTdp+CeaaBquN188JKSkmI345TXu9d2teuqJtjVE7s+BTWTffr06XLtq6++KuNu29TcdVcH7Z7loPocXH24q013z1NQvQau9tx9LvWsBhWL8M88cL0hatvd2q6uLhlX/QK7du2Sa3t7e2V86tSpmTFXr+/OQ/ecCEU98yPCf66RI0dmxlpaWuTazZs3y7g7T1Wvj9vuYnCnAABISAoAgISkAABISAoAgISkAABISAoAgKToklRXJqW48dWubNSVw+YZne3K3tToXzVWOyKitrZWxs8555zMmNsnY8eOlfFt27bJuBqJrMYCR/hyWFWS19DQINc2NjbKeFVVlYy7Ee+KG+utyhD37t0r1+7cuVPGOzs7Zfy3fuu3MmPPP/+8XOvOU1VOPnfuXLl2woQJMv7YY49lxtT+jPDXj7t2Vfmz+15wZfSqpLupqWnA2xXhR52r4+nKqovBnQIAICEpAAASkgIAICEpAAASkgIAICEpAAASkgIAICm6T8HVv6peAjee2tUMu9pzVVPs6v1dzbD6XKrWPyKivb1dxrds2ZIZGzNmjFzrRkyfe+65Mq76L9Qo5Qj/uVV81KhRcm19fb2MO3nOBTeKWY3HduOS1bGO8NumRlhPmTJFrnXXj+pLOf/88+Va119x3nnnZcaGDx8u17pz3J2H6nvF7W83WludK2673Gu771rVM5b3MQQR3CkAAE5AUgAAJCQFAEBCUgAAJCQFAEBCUgAAJCQFAEBSdJ9CnmceuDn1rk/BxRU3F93VcKu56e61VV17RMTBgwczY2vWrJFr3TMmhg4dKuPNzc2ZMdXDEKFrzyMihg0blhlrbW2Va11tunuewnv5bI08fQoTJ06Ucfe51PMY9u/fL9dOmjRJxlW/wH//93/Lte5cmT17dmbM7RPXN+KuXfWdlfe5A+7aV9z3oXt2jTqP3fdCMbhTAAAkJAUAQEJSAAAkJAUAQEJSAAAkJAUAQEJSAAAkv7I+hTxzvPPU/Dqu/rumpkbGVd28exaDi6t6ZLdP3P52NfcjR47MjI0fP16uPXLkiIw3NTVlxtzxyEs9u8PtE0cdr9raWrl28uTJMu7OQ/X6Bw4ckGvdPu/p6cmMNTQ0yLVjx46V8TPPPDMz5s5x1SMUke87xz3jxVHv7foQent7Zdx9bxw+fHjA710M7hQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQFF2S6sa5qlKovOWTjiptq6urk2vVmOcIXZLnSurc51KjgV2JohsbXFFRIeOq3M+NLFajliMiSktLM2NutK/7XHnGqLtz2I1TVsfbjfx254rbL+r11Qj2CF9+qUog3Xnkri+13e5YO+5cUMc7b+mm2qfutd3xctefGleet9Q2gjsFAMAJSAoAgISkAABISAoAgISkAABISAoAgISkAABIiu5TyFPDnafHoZj4kCFDMmOuftyNPFZ9DG60rxuBq+q08/YhqF6BCN1D4faZ679Q9eOuttzVWeepw3bnkesVUNeA68Vxx2Po0KEDXu/OBbdt6tp156G7ftR6t13uXHHr1bniziN3ruTpgXDfh3mOF6OzAQC/UiQFAEBCUgAAJCQFAEBCUgAAJCQFAEBCUgAAJEX3KVRVVcm4mhHuarBd3buLq1po1yugehwidC9Cc3OzXOvqjdW2uc/satPzzNDPS7133s/lnkug3tvVvas59RH6eLrtOnz4sIy7Hgm1X9w+cz1GKt7d3S3X9vT0yLjaL67Pp7e3d8CvHaGPt7s23Xur49XW1ibXOu7aPXLkSGbM7ZNicKcAAEhICgCAhKQAAEhICgCAhKQAAEhICgCApOiS1IaGBhnfs2dPZsyV+rmxwnV1dTLuxvcqEyZMGPBaV+rntkt9blc+6UYau9I0VZLnyvWOHj0q4+pzuWPtRv+6z1VeXp4Zc+WV7nOpuCthVNsV4csQVSmvK2d1n1vtU1eK7rZbXSN5yj6LeW9Vuum+k1yZryo7daOx3TnsPreKu31aDO4UAAAJSQEAkJAUAAAJSQEAkJAUAAAJSQEAkJAUAABJ0X0KbsR0S0tLZuznP/+5XOtq0109sqoBd6/teg1UTfCBAwfkWlfvr2r26+vr5VpXe+6ounfXA9HZ2Tng166urpZrXY22G0Gt6rTz9suofe7q2nfs2CHj7vpS53FXV5dc63ok1D51463d8Whvb8+MuX3meiTc8VT7zF33biS4Wu8eFeB6kNzxVN8r7vuuGNwpAAASkgIAICEpAAASkgIAICEpAAASkgIAICEpAACSovsUxo0bJ+OqZtjVYO/atUvGZ86cKeOzZs3KjLm6d7dt+/fvz4yNGjUq12uree8dHR1ybVNTk4y7fgBVX/6LX/xCrnV11s3NzZmxmpoauTZv74eqEVf9ExG+Nl1tuzvW6nkjEXo+f4Su6Xe9BK7/Ql27rh+msbFRxtV5pq6tCF+v767twYMHZ8Zcj8S+fftkXJ1LqmcrIuKCCy6QcfW9EBExfPjwzJh7JkgxuFMAACQkBQBAQlIAACQkBQBAQlIAACQkBQBAQlIAACRF9yls3LhRxsePH58Zc3Xtrqbe9UisXbs2M+bq4l18xIgRmTFXy+xmtqt65kGDBuV6bbVPIiIaGhoyY6oOupj3VjP0VSwi4qyzzpLxuro6GVe17+54jR49WsZ3794t44p77sDevXtl/Oyzz86Mud4Od7zUszvceej2ieq3cefZli1bZHzs2LEyrrbN9ay480ydSy+88IJc6/oQ3PNMli9fnhmbPHmyXFsM7hQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQFF2SWigUZLy8vDwzdvz4cbnWjYHeuXOnjG/bti0z5kYauzG3ahTzwYMH5Vr33v/zP/+TGZs4caJc68orXSmhGo99zjnnyLWufHLz5s2ZsR07dsi17nO549XX15cZc2WG27dvl3H1uVzZtSsbdSWSqpTXnYduxLQqG1Vj0CMient7ZVyN5VaxCF3mHuFLVtW54MZy19bWyrgq833zzTflWve9sHXrVhlXZdevv/66XFsM7hQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQkBQAAEnRfQqVlZUyXlFRkRlzPQ6qZj4i4vzzz5fxCRMmZMbUWOCIiNLSUhlXNcduTK2ro1bbNnjwYLl21KhRMu76AVS9v6uTHjZs2IDjqqckQtfMR/ixw6o2/Td+4zfkWjcG+tlnn82MqVHkEb5Xx9XNjxw5csCv7XooVE/Lrl275Fp3PNR7d3Z2yrXuGnD7vK2tLTOmvq8i/KjzF198MTPmRoI7U6ZMkXG1z9V3YbG4UwAAJCQFAEBCUgAAJCQFAEBCUgAAJCQFAEBCUgAAJEX3KbS2tsp4T09PZszVSbt65aefflrGP/OZz2TGysrK5NrVq1fLuKoJdjXaI0aMkHFVX65mpkdElJSUyHh1dbWMq2ciuF4BNUs+Qs+id89qcL0drpdg9OjRmbGnnnpKrnX9F+eee25mTPVHRPiaencuqbp590wD96wG93yMPK+trn31nRHh+5fcevUsFNeH4PppnnjiicyY6+nas2ePjLtrQPUxuGdrFIM7BQBAQlIAACQkBQBAQlIAACQkBQBAQlIAACQlBTfX+v9xY4c3bdqUGXPlX66szZVZqTHQbrvd6OyamprMWHl5ea7XVmbNmiXj7e3tMn7gwAEZr6qqyoypUr4I/7nV8XbH0sXdKGdVyltXVyfXqpLTiIhJkyZlxtyxdpeZK2lVo7U3btwo17pSW1U2evrpumrdldKqknA38tuVPrt9qs7Do0ePyrVq7HZExB133JEZy1PiG+E/lzom7tp05f8R3CkAAE5AUgAAJCQFAEBCUgAAJCQFAEBCUgAAJCQFAEBS9OjstWvXyrga3zt48GC51tU6O93d3Zkx1yNRUVEh466WWnG164MGDcqMrV+/Xq519ePuvVVtuqsPd9y4csVtt+tpUXXabnz1qFGjZFxtmzvH3edyte2HDh3KjLlz2F0DatvV+0boczhC91+48e9Ont4PN85/1apVMq7GX7teAXd9uXNF9Rip78JicacAAEhICgCAhKQAAEhICgCAhKQAAEhICgCAhKQAAEiK7lNw9eGq7tfVE7vXdjo6OjJju3fvlmuHDBki42rb3Qx813+h6pldbbnj9ql63oKruXe9G3mOpzqWEf45Eer5F663w+1ztd7VlufpFXCv746H6xtRPUbutd12q+cWuO8F18fgri91ffb09Mi1GzZsGPBru33ieiTcuaK23T0nohjcKQAAEpICACAhKQAAEpICACAhKQAAEpICACApuiTVlV+qEbpurSthzDO+uq2tTcZbWlpkXJXFuc/lRuS69Yor5+vq6pJxdbw6OzvlWld+uXXr1szY6tWr5dpdu3bJeJ7STjfm2Y2gnjZtWmZs9uzZcu3QoUNlXJWFRuhrwG23K91UJZKuvNJRr+3Ofxd314Dapzt27JBrN2/eLOPqeLrSZzeOPE95c95x5BHcKQAATkBSAAAkJAUAQEJSAAAkJAUAQEJSAAAkJAUAQFJ0n4KrvVX1sa4+3NXWuvdWfQ47d+6Ua11d/MSJEzNjeUYtR+g6azcC1/V2uBpvVX/u6t7ffPNNGX/ppZcyY65vpLKyUsbdtqnR226t6yv58Y9/nBlbs2aNXHv55ZfLeGtrq4yrbXPHura2VsZV3bzrEXK9Aurad6+dtw9Ivf5rr70m13Z3d8u4Gkfu+kLyjgxXcfoUAAC/UiQFAEBCUgAAJCQFAEBCUgAAJCQFAEBCUgAAJEX3Kbj6V1V762rPXb2yqxnOU7e7bds2GW9ubs6MVVdXy7WOqrN2c+zdPqmqqpJxVbO/ceNGudY9E2HdunWZsf3798u1qv47wtfcq/4ON6feOXjwYGZs+/btcq3qn4iIuOKKK2R82LBhmTE3n7+8vHzAcffaeXqQ1LMWInw9v4vv2bMnM7Zq1Sq51vVA5PnOcccjT5+C640qBncKAICEpAAASEgKAICEpAAASEgKAICEpAAASEgKAICk6KJWVzevZoi7ml9X6+zeW9VSu+cOuOcpqPrzyZMnD3i7InTdfFdXl1zr6vVdD8WOHTsyY/v27ZNrN2/eLOOqJt89g8LN0HfrVZ9DZ2enXOvq5lU/jVv7zDPPyLj73L/927+dGXN9QL29vTKu6ubda/f09Mh4HnmelxAR8fLLL2fG3HWf5xkv7li67ztHXQOuP6kY3CkAABKSAgAgISkAABKSAgAgISkAABKSAgAgKbok1Y2pVeWVrnTMlfO5slJV+um221GjtUeNGiXXurJRVVo2ZMgQudaNmN69e7eMq7I4V47nykJViWPec8GNBlal0e61XSmh4kozXVnomjVrZPxjH/tYZsyNBHdxdf24febKxdXnzlty6sbev/jiiwN+b/edo87DPI8ZyLveHa9icKcAAEhICgCAhKQAAEhICgCAhKQAAEhICgCAhKQAAEiK7lNwtbOqxtuNinU1wa6uV71+3rp4Ve+/Z88eudaNHVb1467+u62tbcCv7dTX18v4yJEjZfytt97KjKkehmKoPoQIfS6488z1X6hzyfWNuPNw3LhxMj58+PDMmLs+3OdS+yXva6vzsLu7W651++w//uM/ZPzAgQMD2q5iqO9Dt93uvV1PizrX8vZlRXCnAAA4AUkBAJCQFAAACUkBAJCQFAAACUkBAJCQFAAASdF9ChUVFTKuZrKrWDFczb6Ku5pf16eg6pF/8YtfyLWtra0yruqNXR+C6/3IO9NdGT16tIyr471582a51m2X63NQNeJun7h4VVVVZqy9vV2uHT9+vIxfeOGFA35v17uR53O768fV5KvnFqjPFBHx+OOPy/jrr78u4+qZB11dXXKt69VR1+7Ro0flWsf106j+C/ddWQzuFAAACUkBAJCQFAAACUkBAJCQFAAACUkBAJAUXZLqSgVV+ZcrsVJjt91rR0RUV1dnxly5nour8jI3Ltl9rl27dmXG3Gd2Y7nd51Lb7o6XK3ubOnVqZsyVAK9du1bG3edS2+7KJ13pZkdHR2Zs2rRpcu1ll10m483NzTKuuHPFnYfqHHdjnt3xUKO11f6MiFi/fr2MOz09PZkx9Z0R4c9Tt0/zvLbbL+p4u3O8GNwpAAASkgIAICEpAAASkgIAICEpAAASkgIAICEpAACSovsU1AjcCF176+r589aPqxG8bvSvG+utRui68bpvvPGGjKvR2nV1dXJtd3e3jLv6clXPr2rLI/zxVO89a9Ysudbt01dffVXGVY23qw93vR8XX3xxZmzu3LlyrRs973o/VG163vHw6vpza933gupvevTRR+XaTZs2yfi4ceNkfN26dZkx953jznHVA+H6Rhx3PFWPhRupXwzuFAAACUkBAJCQFAAACUkBAJCQFAAACUkBAJCQFAAASb6C2hOoul5V0xvhn9Xg+hSU8vJyGXdz7NX6LVu2yLXDhg2T8f3792fG8ta1u3nvqhfBzZp3ddg7d+7MjDU0NMi1s2fPlvGZM2fK+IsvvpgZc+fZeeedJ+NDhgzJjOV5zkOEP95dXV2ZMVebnmf2v6vnd/0wP/vZzzJjW7dulWtdL45z1llnZcbUs0wifK+AOpfctamOZYT/zlLceVYM7hQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQkBQAAEnRfQquDlvVWbvZ5Hlntqu6+eHDhw94bUTECy+8kBlzNfeOqh8/ePCgXOvqkV2ttHqOhHuegnvWg6rxdr0dnZ2dMt7S0iLjl156aWbM1Ye791bHxD2Lwe0zd32pXgTXf+GuL3Ueuh6Ibdu2yfj69eszY+o5KBH+PHTHU/U3uWcxbNiwQcZVf4brq3LHy8XVtZ+nJ+Vt3CkAABKSAgAgISkAABKSAgAgISkAABKSAgAgKbok1ZVuqjJENwrWlWCp8smIiPr6+sxYTU2NXPvaa6/J+NChQzNjbqywKw9T648ePSrXunI9x5VQKh0dHTKuRoZv2rRJrm1ra5NxNZY7Qp8LruzTnWfqGpg6dapc646nO5fUe7sSSBdX5+mBAwfkWnc8VRmv+15wpbTuO6m9vT0z5srk1djtCH2euhJ697lcGbBa70aZF4M7BQBAQlIAACQkBQBAQlIAACQkBQBAQlIAACQkBQBAUlJwTQIAgI8M7hQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQkBQAAAlJAQCQkBQAAMn/ARyuuTSC82mBAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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CUkBAJCQFAAACUkBAJAUXZLqSp2icj43dtuN9nUlqXlGULtti9bnLWvLUwp4IscKu2Piyiej/XIjpvOMxpbic8WVH7tzPDqme/bsCde6klRXDhuNanaftTvm0eeVZ7T8iZbnvaPjKeUrk8973bt4NM58586d4dpu3bqFcYk7BQDAEUgKAICEpAAASEgKAICEpAAASEgKAICEpAAASIruU3B11ps2bcqMufpwV3PvatejGm9X8+vq3qM6bVf/nWescJ61xcQjeeuoo+PiPku33507dw7jUX15RUVFuNadp9G2RSO7JamtrS2M19bWhvGoNn3VqlXhWlebHh0z93m4eJ7x164/yb139Hnm3a+I2+48vVFS/H23YcOGcG2/fv3CuMSdAgDgCCQFAEBCUgAAJCQFAEBCUgAAJCQFAEBCUgAAJEX3Kbia/Pr6+szYu+++G651tczuvaN4ntnk7rVPZJ9Cnj4DyW9bnh4Jx/Ui5FnraryjuOsVcM80iJ63kPfzcuu7d++eGVu+fHm4dseOHWE8unbz9KRI8bnknlmQ972j75WysrJwrfs83HtHXB/Dx3ntStwpAACOQFIAACQkBQBAQlIAACQkBQBAQlIAACQkBQBAUnSfwv79+8N4VVVVZqyxsTFc6+rH3az6qJ4579z0qCbY1RO7OusTtVbKV68c1eNLftui+nPXh+Be221bVJuep65dis9T95wH90wDd1yi66uhoSFcu3bt2jDetWvXMB5xvQau9yOSp99FirctT5/BiX5t14sTvf6AAQNyvbfEnQIA4AgkBQBAQlIAACQkBQBAQlIAACQkBQBAUnRJqhslG5WPRaN5JWnfvn1h3JUpRnG33S0tLWE8KjXMW+4abbfb57wjcqOytvfeey9c60b/RmWjrsxw7ty5YXz16tVh/KqrrsqMuTHpruw6Ohd2794drv35z38exl3J6mWXXZYZyzPyW4pLcV3JqYvnHSme572j89SVfeZ5bXdtumvAnafRd0NFRUW4thjcKQAAEpICACAhKQAAEpICACAhKQAAEpICACAhKQAAkqL7FFx9+PDhwzNjeUfJurreqH7c1dy7euU8NdyuRjtPn0Je0eu7Y+JGnUc13K4npWfPnmF869atYXzVqlWZscGDB4drnaj+fM6cOeHaBx54IIzfdNNNYTw6bjt37gzX5hlB7XpSXE19Hu76yXP95RlPLcXfK2VlZeFaN6LdbVsUb25uDtdWVlaGcYk7BQDAEUgKAICEpAAASEgKAICEpAAASEgKAICEpAAASIruU5g1a1YY79+/f2bM1e3mqeeX4pphV8vsZp9HNcUnclZ83j6FPM+gcHXtbr+jY+ZqsOvq6sL4pZdeGsajz9P1rJSXl4fx6Nkbp512Wrj2uuuuC+NDhw497vd2telVVVVhPPo83bMaXD1/9Hm47wV3HroeijzPHMnTI9Ha2hqura6uPu7XluLnY7hjWgzuFAAACUkBAJCQFAAACUkBAJCQFAAACUkBAJCUFIqsffzSl74UxkeNGpUZ+7M/+7Nw7Z49e8J4VIIlxaWErgQyT/mlK3nLO1o74koB3ccavbcb7es+j2icshud7USlmVJ8Lrj3dvsVrc9Tpiv5ctnoXHPXjyvdjMoY3ajlPCXfrnzSneOuXDZ67zwj86V4v/OW4LuS1YqKiszYvHnzwrVjxowJ4xJ3CgCAI5AUAAAJSQEAkJAUAAAJSQEAkJAUAAAJSQEAkBQ9Oru+vj6ML1iwIDPmaub/4A/+IIy79fv378+MuVrmzp07h/Goj8HVUTtRHbXrgXBxd8yibXc183lq013de/RZSlKXLl2O+73dfrl4VD/e1tYWrnXnmTuX9u7dmxnr1q1buNZdA9Exd8fE1dxH/QDuHHa9BHn6gFx/ktu2aL17bXcuuHP86aefzow98MAD4Vr6FAAAHwpJAQCQkBQAAAlJAQCQkBQAAAlJAQCQkBQAAEnRfQqufryhoSEz9uKLL4Zr16xZE8anTJkSxqN6Zlf/7WboR/XKeZ6HIMX1zO618/QKuPd2r+2OWbTe1XC72f/uuOQ5ptGceinu/cizXZLf7+iYurXuvaP9cs8VcPJcI+48zPPerlfA9Z24Yx5xvTq7du0K448++mhmzPV2FIM7BQBAQlIAACQkBQBAQlIAACQkBQBAQlIAACRF1y+50rKoBLK2tjZcu3LlyjC+ZMmSMH7OOedkxlpbW8O1roQrz3hrJyqXzTuWOw9XflxWVhbGoxJHd7xdCaQrU4yOm1vr3jsqQ3Qjpp087+3OFTc6O3ptdy6474U8Jal5Xzs6D91+OdHnVVVVFa4tLy8P4/Pnzw/je/bsyYyVlpaGa4vBnQIAICEpAAASkgIAICEpAAASkgIAICEpAAASkgIAICm6T8HVQkd12q5u19UML1y4MIyPGjUqM5ZnhLQU1/3m7SXIU8Odd6RxdFzy1vNHXI+Ds3nz5jCe5zNxNd7RfufpM5D8cYnGlecZjS3F2+b2y713JG8fQp4+hbxj7/Ocx+69X3vttTAe9fq4npRicKcAAEhICgCAhKQAAEhICgCAhKQAAEhICgCAhKQAAEiK7lNw9cpRjXdzc3O41s0XX79+fRhfsWJFZmzw4MHhWldnnafuPc8zKFxtuZNnu12fQp7nSESz4CXplVdeCeOrV68O41E9f8+ePcO13bt3D+Pbtm3LjLlztLKyMoyPHj06jNfX12fG3OfleiQi7rWj4+3k7UM4kf00TnQNuL4sdw6vXbs2jEfPp3HXVzG4UwAAJCQFAEBCUgAAJCQFAEBCUgAAJCQFAEBCUgAAJB9Zn0JLS0tmLM+seMn3Erz00kuZMVeb7mq4o3rkaK65Wyvl60XIM0teip9/kfe1o5nurg9h0aJFYTzPMw+WLl0arnX14fv27cuMuZp5Vz/+3HPPhfEvfvGLmbHevXuHa93zSqJrwJ3jebjrI08/jJTveQruuQTRcencuXO41j0vwfV1RZ9nnp6Uw7hTAAAkJAUAQEJSAAAkJAUAQEJSAAAkJAUAQPKR1ZtFJY5RrBiufKy1tTUztnfv3nDtgAEDwvju3bszY2489Ykcne3i7r3zrHXll1Fp55o1a8K17pi++uqrYfzyyy/PjJ1++unh2nXr1oXx6Li4smk3Ljkq6ZakX//615mxHj16hGvzlCm688ydCxFXcprnHHbr3X65Uty6urrM2LJly8K1s2bNyvXeUYl/nlHmh3GnAABISAoAgISkAABISAoAgISkAABISAoAgISkAABIiu5TcKN/Kyoqjism+dG//fv3D+NRzXHUZyD5keBRLXU0SlmSKisrj/u984zVLma9q6vP89o7duzIjO3atStc++yzz4Zx1+ewadOmzNill14arl2yZEkYb2xszIxVV1eHaxcuXBjGXa9BU1NTZmzFihXh2hEjRoTx6BpxY6DdeRTV3Luaejcm3fW0ROep65Fwo7Oj8dUzZ84M127ZsiWMu96PaNvcMSkGdwoAgISkAABISAoAgISkAABISAoAgISkAABISAoAgKToPoVTTjkljPft2zczVl9fH2+EmR/u6uKjut/FixeHa109ctQj4WqdXQ139JwJ19vhuOdIRDP23Xa7+vKNGzdmxubOnRuudc80aGhoCOORpUuXhvFf/vKXx/3ebvZ/nz59wnjUXyHF/R0rV64M17p6/2i/XB+Pe1ZKdG2Xl5eHa6NeAMlff1G9f956/hdeeCEz9vrrr4dr3X676++j6EWIcKcAAEhICgCAhKQAAEhICgCAhKQAAEhICgCApOiS1DPPPDOMR2Ok3ajYaNRyMfHovd3az372s2HcleJGXCltNJbYlTi6sd2uVLBr166ZMXfM3DjyqJTQlRmeccYZYbxXr15hvKWlJTPmRpmPHTs2jEeGDRsWxt3nkack1Y3ddlpbWzNjXbp0Cde6stDoGigrKzvu7ZL8NRKVm7u1bsR7VFrt9ivPuH4pLlnNO3Jf4k4BAHAEkgIAICEpAAASkgIAICEpAAASkgIAICEpAACSovsUXG1tNPLYjUN2Y57dGOmorjfqBZCk+fPnh/Go/tzVvbsRt1G9cVtbW7i2qqoqjLta6ebm5sxYNHJY8j0Szz//fGbM1fO7sdxu9Hbv3r0zY1deeWW49le/+lUYjz4v14ewZMmSMH7WWWeF8ei4zJkzJ1zrztNu3bplxqK+D8mfK9G163oB3Mhvd21HPTHReSJJ//Ef/xHGt27dmhmrra0N17o+H9drEMXd93QxuFMAACQkBQBAQlIAACQkBQBAQlIAACQkBQBAQlIAACRF9ym4Gd/bt2/PjLk+hI4dOxa7GR8omrve1NQUrnU1w9Gc+8GDBx/3dkm+Djvinkvg5sVHxzyqx5ekmpqaMD569OjM2KuvvhqujWbgS1L//v3DeH19fWbM1XB/5jOfCeNRL4LrBXDb7c6FaNtdXbzbtqju3Z1HrqY+2m73Wed5roAU73fUpyNJr7zyShiPenVcz4rrIXLXttvvvLhTAAAkJAUAQEJSAAAkJAUAQEJSAAAkJAUAQEJSAAAkRfcpuHrkiKtHPvnkeDNcffnAgQMzY8OHDw/Xrl69OoyvXLkyMzZkyJBwrTtmUT2ymxXvuFrp6FkPrr+iuro6jJ9xxhmZsdmzZ4drd+zYEcYvvPDCMB71nbz11lvhWtcrEPUa1NXVhWujZxZI0ooVK8J41OszaNCgcK3rU4iea5C3ByKqqXfPG8n7vRH1AyxdujRc60TX5549e8K1br9dPOrfcGuLwZ0CACAhKQAAEpICACAhKQAAEpICACAhKQAAkqJLUp2o/NKVlOYpLZPiUsG2trZw7c6dO487fuDAgXCt2+6obDTv2GA38jhSVVUVxt1+R2OFo3JVyY8sdvFevXplxvKMkJbi/XKlgKtWrQrj7vOK4u76caOYu3btmhnLU1Ytxce8Q4cO4VpX+uzOw2g8/MaNG8O17nsj+rxdObnbbnftR/vlStGLwZ0CACAhKQAAEpICACAhKQAAEpICACAhKQAAEpICACApKeSZiQ0A+FThTgEAkJAUAAAJSQEAkJAUAAAJSQEAkJAUAAAJSQEAkJAUAAAJSQEAkPw/WNt+jwFShfYAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "# STEP 3: Set Dataset Paths\n",
        "train_data = '/kaggle/input/fer2013/train'\n",
        "test_data = '/kaggle/input/fer2013/test'\n",
        "\n",
        "\n",
        "\n",
        "#priview the images\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import os\n",
        "from PIL import Image\n",
        "\n",
        "# Manually pick a class and image\n",
        "sample_class = os.listdir(train_data)[0]  # e.g., 'Surprise' for 0 index\n",
        "sample_class_path = os.path.join(train_data, sample_class)\n",
        "\n",
        "# List image files inside the class folder\n",
        "image_files = os.listdir(sample_class_path)\n",
        "\n",
        "# Load and display the first 2 images\n",
        "for i in range(2):\n",
        "    img_path = os.path.join(sample_class_path, image_files[i])\n",
        "    img = Image.open(img_path).convert('L')  # convert to grayscale\n",
        "    plt.imshow(img, cmap='gray')\n",
        "    plt.title(f\"From class: {sample_class}\")\n",
        "    plt.axis('off')\n",
        "    plt.show()\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# STEP 4: Data Generators\n",
        "train_datagen = ImageDataGenerator(\n",
        "    rescale=1./255,\n",
        "    rotation_range=30,\n",
        "    shear_range=0.3,\n",
        "    zoom_range=0.3,\n",
        "    horizontal_flip=True,\n",
        "    fill_mode='nearest'\n",
        ")\n",
        "\n",
        "validation_datagen = ImageDataGenerator(rescale=1./255)\n",
        "\n",
        "train_generator = train_datagen.flow_from_directory(\n",
        "    train_data,\n",
        "    color_mode='grayscale',\n",
        "    target_size=(48, 48),\n",
        "    batch_size=32,\n",
        "    class_mode='categorical',\n",
        "    shuffle=True\n",
        ")\n",
        "\n",
        "validation_generator = validation_datagen.flow_from_directory(\n",
        "    test_data,\n",
        "    color_mode='grayscale',\n",
        "    target_size=(48, 48),\n",
        "    batch_size=32,\n",
        "    class_mode='categorical',\n",
        "    shuffle=True\n",
        ")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OfjwfiETa6sJ",
        "outputId": "58c40897-96a4-4810-98bc-85c2294d4cad"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 28709 images belonging to 7 classes.\n",
            "Found 7178 images belonging to 7 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# STEP 5: Model Definition\n",
        "model = Sequential()\n",
        "\n",
        "model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))\n",
        "model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))\n",
        "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "model.add(Dropout(0.1))\n",
        "\n",
        "model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))\n",
        "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "model.add(Dropout(0.1))\n",
        "\n",
        "model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))\n",
        "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "model.add(Dropout(0.1))\n",
        "\n",
        "model.add(Flatten())\n",
        "model.add(Dense(512, activation='relu'))\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Dense(7, activation='softmax'))  # 7 output classes\n",
        "\n",
        "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
        "\n",
        "model.summary()\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 632
        },
        "id": "SYSLocmxa_ED",
        "outputId": "76db4d57-ef26-4625-c281-043ffbabdf88"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1mModel: \"sequential\"\u001b[0m\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m46\u001b[0m, \u001b[38;5;34m46\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │           \u001b[38;5;34m320\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m44\u001b[0m, \u001b[38;5;34m44\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │        \u001b[38;5;34m18,496\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m22\u001b[0m, \u001b[38;5;34m22\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout (\u001b[38;5;33mDropout\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m22\u001b[0m, \u001b[38;5;34m22\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m20\u001b[0m, \u001b[38;5;34m20\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │        \u001b[38;5;34m73,856\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m10\u001b[0m, \u001b[38;5;34m10\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout_1 (\u001b[38;5;33mDropout\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m10\u001b[0m, \u001b[38;5;34m10\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_3 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m8\u001b[0m, \u001b[38;5;34m8\u001b[0m, \u001b[38;5;34m256\u001b[0m)      │       \u001b[38;5;34m295,168\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m256\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout_2 (\u001b[38;5;33mDropout\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m256\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4096\u001b[0m)           │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m512\u001b[0m)            │     \u001b[38;5;34m2,097,664\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout_3 (\u001b[38;5;33mDropout\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m512\u001b[0m)            │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m)              │         \u001b[38;5;34m3,591\u001b[0m │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">46</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">46</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │           <span style=\"color: #00af00; text-decoration-color: #00af00\">320</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">44</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">44</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │        <span style=\"color: #00af00; text-decoration-color: #00af00\">18,496</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">22</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">22</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">22</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">22</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">20</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">20</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │        <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)      │       <span style=\"color: #00af00; text-decoration-color: #00af00\">295,168</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4096</span>)           │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)            │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,097,664</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)            │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>)              │         <span style=\"color: #00af00; text-decoration-color: #00af00\">3,591</span> │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m2,489,095\u001b[0m (9.50 MB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">2,489,095</span> (9.50 MB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m2,489,095\u001b[0m (9.50 MB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">2,489,095</span> (9.50 MB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# STEP 6: Count Number of Images\n",
        "def count_images(folder_path):\n",
        "    return sum(len(files) for _, _, files in walk(folder_path))\n",
        "\n",
        "num_train_imgs = count_images(train_data)\n",
        "num_test_imgs = count_images(test_data)\n",
        "\n",
        "print(\"Number of training images:\", num_train_imgs)\n",
        "print(\"Number of test images:\", num_test_imgs)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z4gAaOhEbFkw",
        "outputId": "3449d965-ba96-4746-c04d-dbb1d9934884"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Number of training images: 28709\n",
            "Number of test images: 7178\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# STEP 7: Train the Model\n",
        "epochs = 20\n",
        "history = model.fit(\n",
        "    train_generator,\n",
        "    steps_per_epoch=num_train_imgs // 32,\n",
        "    epochs=epochs,\n",
        "    validation_data=validation_generator,\n",
        "    validation_steps=num_test_imgs // 32\n",
        ")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Zk3S2RhZbJ6X",
        "outputId": "f839dbe0-a3fa-4b59-d862-a019f321ba36"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
            "  self._warn_if_super_not_called()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 117ms/step - accuracy: 0.2539 - loss: 1.8124"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
            "  self._warn_if_super_not_called()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m140s\u001b[0m 147ms/step - accuracy: 0.2539 - loss: 1.8123 - val_accuracy: 0.3126 - val_loss: 1.6925\n",
            "Epoch 2/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m14s\u001b[0m 16ms/step - accuracy: 0.1875 - loss: 1.8695"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 6ms/step - accuracy: 0.1875 - loss: 1.8695 - val_accuracy: 0.3136 - val_loss: 1.6888\n",
            "Epoch 3/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m99s\u001b[0m 45ms/step - accuracy: 0.3142 - loss: 1.6953 - val_accuracy: 0.4160 - val_loss: 1.5067\n",
            "Epoch 4/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m12s\u001b[0m 14ms/step - accuracy: 0.2500 - loss: 1.7711"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 6ms/step - accuracy: 0.2500 - loss: 1.7711 - val_accuracy: 0.4149 - val_loss: 1.5082\n",
            "Epoch 5/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 45ms/step - accuracy: 0.3842 - loss: 1.5684 - val_accuracy: 0.4572 - val_loss: 1.3979\n",
            "Epoch 6/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m13s\u001b[0m 15ms/step - accuracy: 0.2812 - loss: 1.5800"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 6ms/step - accuracy: 0.2812 - loss: 1.5800 - val_accuracy: 0.4566 - val_loss: 1.3975\n",
            "Epoch 7/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 46ms/step - accuracy: 0.4140 - loss: 1.4940 - val_accuracy: 0.4587 - val_loss: 1.3936\n",
            "Epoch 8/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m12s\u001b[0m 14ms/step - accuracy: 0.3125 - loss: 1.5695"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 6ms/step - accuracy: 0.3125 - loss: 1.5695 - val_accuracy: 0.4601 - val_loss: 1.3844\n",
            "Epoch 9/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m76s\u001b[0m 45ms/step - accuracy: 0.4510 - loss: 1.4185 - val_accuracy: 0.5116 - val_loss: 1.2564\n",
            "Epoch 10/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m12s\u001b[0m 14ms/step - accuracy: 0.4375 - loss: 1.4114"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 6ms/step - accuracy: 0.4375 - loss: 1.4114 - val_accuracy: 0.5120 - val_loss: 1.2572\n",
            "Epoch 11/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 46ms/step - accuracy: 0.4763 - loss: 1.3735 - val_accuracy: 0.5361 - val_loss: 1.2147\n",
            "Epoch 12/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m14s\u001b[0m 16ms/step - accuracy: 0.5000 - loss: 1.3499"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 6ms/step - accuracy: 0.5000 - loss: 1.3499 - val_accuracy: 0.5336 - val_loss: 1.2164\n",
            "Epoch 13/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m82s\u001b[0m 51ms/step - accuracy: 0.4877 - loss: 1.3371 - val_accuracy: 0.5321 - val_loss: 1.2144\n",
            "Epoch 14/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m13s\u001b[0m 15ms/step - accuracy: 0.5312 - loss: 1.3906"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 6ms/step - accuracy: 0.5312 - loss: 1.3906 - val_accuracy: 0.5329 - val_loss: 1.2139\n",
            "Epoch 15/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 46ms/step - accuracy: 0.4980 - loss: 1.3164 - val_accuracy: 0.5434 - val_loss: 1.1883\n",
            "Epoch 16/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m12s\u001b[0m 14ms/step - accuracy: 0.6562 - loss: 0.9885"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 6ms/step - accuracy: 0.6562 - loss: 0.9885 - val_accuracy: 0.5455 - val_loss: 1.1892\n",
            "Epoch 17/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m40s\u001b[0m 45ms/step - accuracy: 0.5064 - loss: 1.2956 - val_accuracy: 0.5490 - val_loss: 1.1769\n",
            "Epoch 18/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m12s\u001b[0m 14ms/step - accuracy: 0.5625 - loss: 1.1502"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 6ms/step - accuracy: 0.5625 - loss: 1.1502 - val_accuracy: 0.5490 - val_loss: 1.1792\n",
            "Epoch 19/20\n",
            "\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m40s\u001b[0m 44ms/step - accuracy: 0.5066 - loss: 1.2804 - val_accuracy: 0.5576 - val_loss: 1.1619\n",
            "Epoch 20/20\n",
            "\u001b[1m  1/897\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m12s\u001b[0m 14ms/step - accuracy: 0.4062 - loss: 1.4053"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
            "  self._interrupted_warning()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\u001b[1m897/897\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 5ms/step - accuracy: 0.4062 - loss: 1.4053 - val_accuracy: 0.5585 - val_loss: 1.1598\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# STEP 8: Save the Trained Model\n",
        "model.save('/content/model_file.h5')"
      ],
      "metadata": {
        "id": "WG-ieS3SbQfO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# STEP 9: Plot Training and Validation Accuracy & Loss\n",
        "plt.figure(figsize=(12, 5))\n",
        "\n",
        "# Accuracy Plot\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot(history.history['accuracy'], label='Train Acc')\n",
        "plt.plot(history.history['val_accuracy'], label='Val Acc')\n",
        "plt.title('Model Accuracy')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.legend()\n",
        "\n",
        "# Loss Plot\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.plot(history.history['loss'], label='Train Loss')\n",
        "plt.plot(history.history['val_loss'], label='Val Loss')\n",
        "plt.title('Model Loss')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Loss')\n",
        "plt.legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "9KPsx3aybTHc"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
