"""
Medication Indication Database
Provides information about common medication purposes and usage
"""

# Common medication indications database
MEDICATION_DATABASE = {
    # Antibiotics
    'amoxicillin': {
        'category': 'Antibiotic (Penicillin)',
        'indication': 'Treats bacterial infections (respiratory, ear, urinary tract, skin)',
        'class': 'Beta-lactam antibiotic'
    },
    'moxclay': {
        'category': 'Antibiotic (Amoxicillin + Clavulanate)',
        'indication': 'Treats bacterial infections resistant to plain amoxicillin',
        'class': 'Beta-lactam antibiotic with beta-lactamase inhibitor'
    },
    'augmentin': {
        'category': 'Antibiotic (Amoxicillin + Clavulanate)',
        'indication': 'Treats bacterial infections (sinus, pneumonia, ear, skin)',
        'class': 'Broad-spectrum antibiotic'
    },
    'azithromycin': {
        'category': 'Antibiotic (Macrolide)',
        'indication': 'Treats bacterial infections (respiratory, skin, sexually transmitted)',
        'class': 'Macrolide antibiotic'
    },
    
    # Cough & Cold Medications
    'griinctus': {
        'category': 'Cough Suppressant',
        'indication': 'Suppresses cough and loosens mucus in respiratory infections',
        'class': 'Antitussive/Expectorant'
    },
    'dextromethorphan': {
        'category': 'Cough Suppressant',
        'indication': 'Relieves dry, persistent cough',
        'class': 'Antitussive'
    },
    'guaifenesin': {
        'category': 'Expectorant',
        'indication': 'Loosens mucus and makes cough more productive',
        'class': 'Expectorant'
    },
    
    # Antiparasitic
    'hetrazan': {
        'category': 'Antiparasitic (Diethylcarbamazine)',
        'indication': 'Treats parasitic worm infections (filariasis, elephantiasis)',
        'class': 'Anthelmintic'
    },
    'albendazole': {
        'category': 'Antiparasitic',
        'indication': 'Treats intestinal parasitic worm infections',
        'class': 'Anthelmintic'
    },
    
    # Antihistamines & Allergy
    'awegram': {
        'category': 'Antihistamine + Leukotriene Antagonist',
        'indication': 'Treats allergic rhinitis, seasonal allergies, and asthma symptoms',
        'class': 'Combination allergy medication (Fexofenadine + Montelukast)'
    },
    'smarest': {
        'category': 'Cold & Flu Relief',
        'indication': 'Relieves cold/flu symptoms (congestion, pain, fever, runny nose)',
        'class': 'Combination medication (Antihistamine + Decongestant + Analgesic)'
    },
    'cetirizine': {
        'category': 'Antihistamine',
        'indication': 'Treats allergic reactions, hay fever, hives',
        'class': 'Second-generation antihistamine'
    },
    'fexofenadine': {
        'category': 'Antihistamine',
        'indication': 'Treats seasonal allergies without causing drowsiness',
        'class': 'Non-sedating antihistamine'
    },
    
    # Pain & Fever
    'paracetamol': {
        'category': 'Analgesic/Antipyretic',
        'indication': 'Relieves pain and reduces fever',
        'class': 'Non-opioid analgesic'
    },
    'acetaminophen': {
        'category': 'Analgesic/Antipyretic',
        'indication': 'Relieves pain and reduces fever',
        'class': 'Non-opioid analgesic'
    },
    'ibuprofen': {
        'category': 'NSAID',
        'indication': 'Reduces pain, fever, and inflammation',
        'class': 'Nonsteroidal anti-inflammatory drug'
    },
    
    # Cardiovascular
    'lisinopril': {
        'category': 'ACE Inhibitor',
        'indication': 'Treats high blood pressure and heart failure',
        'class': 'Antihypertensive'
    },
    'amlodipine': {
        'category': 'Calcium Channel Blocker',
        'indication': 'Treats high blood pressure and angina',
        'class': 'Antihypertensive'
    },
    'atenolol': {
        'category': 'Beta Blocker',
        'indication': 'Treats high blood pressure, angina, and irregular heartbeat',
        'class': 'Antihypertensive'
    },
    
    # Diabetes
    'metformin': {
        'category': 'Antidiabetic',
        'indication': 'Controls blood sugar in type 2 diabetes',
        'class': 'Biguanide'
    },
    'glimepiride': {
        'category': 'Antidiabetic',
        'indication': 'Lowers blood sugar in type 2 diabetes',
        'class': 'Sulfonylurea'
    },
    
    # Gastrointestinal
    'omeprazole': {
        'category': 'Proton Pump Inhibitor',
        'indication': 'Reduces stomach acid for heartburn, ulcers, GERD',
        'class': 'PPI'
    },
    'pantoprazole': {
        'category': 'Proton Pump Inhibitor',
        'indication': 'Treats acid reflux and stomach ulcers',
        'class': 'PPI'
    },
    
    # Respiratory
    'montelukast': {
        'category': 'Leukotriene Antagonist',
        'indication': 'Prevents asthma attacks and treats allergic rhinitis',
        'class': 'Anti-asthmatic'
    },
    'salbutamol': {
        'category': 'Bronchodilator',
        'indication': 'Opens airways during asthma attack or breathing difficulty',
        'class': 'Beta-2 agonist'
    },
}

def get_medication_info(medication_name: str) -> dict:
    """
    Get medication information from database
    
    Args:
        medication_name: Name of medication (case-insensitive)
        
    Returns:
        Dictionary with category, indication, and class
    """
    # Normalize medication name
    med_name = medication_name.lower().strip()
    
    # Remove common suffixes
    for suffix in [' tablet', ' syrup', ' capsule', ' mg', 'mg', 'mcg']:
        med_name = med_name.replace(suffix, '')
    
    # Remove dosage numbers
    import re
    med_name = re.sub(r'\d+', '', med_name).strip()
    
    # Look up in database
    if med_name in MEDICATION_DATABASE:
        return MEDICATION_DATABASE[med_name]
    
    # Try partial match
    for key in MEDICATION_DATABASE:
        if key in med_name or med_name in key:
            return MEDICATION_DATABASE[key]
    
    # Return generic info if not found
    return {
        'category': 'Medication',
        'indication': 'Consult pharmacist or physician for specific indication',
        'class': 'Not in database'
    }

def enhance_medication_list(medications: list) -> list:
    """
    Enhance medication list with indication information
    
    Args:
        medications: List of medication strings or dicts
        
    Returns:
        Enhanced list with indication information
    """
    enhanced = []
    
    for med in medications:
        if isinstance(med, dict):
            # Already in dict format
            if 'indication' not in med or not med['indication']:
                # Add indication from database
                info = get_medication_info(med.get('name', ''))
                med['indication'] = info['indication']
                med['category'] = info['category']
            enhanced.append(med)
        else:
            # String format - parse and enhance
            parts = med.split('-')
            name = parts[0].strip() if parts else med
            dosage = parts[1].strip() if len(parts) > 1 else 'N/A'
            frequency = parts[2].strip() if len(parts) > 2 else 'N/A'
            
            info = get_medication_info(name)
            
            enhanced.append({
                'name': name,
                'dosage': dosage,
                'frequency': frequency,
                'indication': info['indication'],
                'category': info['category'],
                'full_text': med
            })
    
    return enhanced


# Example usage
if __name__ == "__main__":
    # Test medication lookup
    test_meds = [
        "Moxclay 525mg",
        "Griinctus syrup",
        "Hetrazan 100mg",
        "AweGram tablet",
        "Smarest tablet"
    ]
    
    print("Medication Information Lookup Test")
    print("=" * 60)
    
    for med in test_meds:
        info = get_medication_info(med)
        print(f"\n{med}:")
        print(f"  Category: {info['category']}")
        print(f"  Indication: {info['indication']}")
        print(f"  Class: {info['class']}")
