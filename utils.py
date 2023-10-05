class bcolors:
    WHITE = '\033[97m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_kmer_sequence(sequence: str, kmer_val: int = 3) -> str:
        """Constructs kmer sequence from full sequence. Divides the DNA/RNA elements into overlapping
        substrings of a fixed window value.
        Ex. ABCD -> kmer=2 -> AB BC CD (3 substrings with length=kmer=2)
        Args:
            sequence (str): RNA sequence
            kmer_val (int, optional): Kmer window length. Defaults to 6.
        Returns:
            kmer_sequence (str)
        """
        kmer_sequence = ""
        sequence = sequence.lstrip()
        for i in range(len(sequence) - kmer_val + 1):
            substring = sequence[i:i+kmer_val]
            if len(substring) == kmer_val:
                kmer_sequence += (substring.strip() + " ")
            else:
                continue
        return kmer_sequence