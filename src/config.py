dt = 0.001

PONG_V4_PAR = {'dt' : dt,
       'tau_m' : 6 * dt,
       'tau_s' : 4 * dt,
       'tau_ro' : 1*dt,
       'tau_star' : dt,

    	'N' : 500, 'T' : 800, 'I' : 3, 'O' : 3,

	'dv' : 0.05 , 'Vo' : -4, 'h' : -8, 's_inh' : 100,

       'gamma' : .99,
       'lerp'  : 0.01,

       'sigma_Jrec' : 0.0,
       'sigma_Jout' : 0.001,

       'alpha' : .01,#0.1*10*.5*.5*.5*,
       'alpha_rout' : 0.0005,#0.01*.5*.5*.5,

	   'sigma_input' : 10.,
       'sigma_teach' : 10.,
       'hidden_steps' : 1,

       'policy_thr_tau' : 1,

       'outsig' : True,
       'step_mode' : 'amax',

       'epochs'     : 0,
       'epochs_out' : 0,

       'clump'    : False,
       'validate' : False,
       'feedback' : 'diagonal',
       'verbose'  : True,
       'rank'     : None
}

PONG_V4_PAR_I4 = {'dt' : dt,
       'tau_m' : 6 * dt,
       'tau_s' : 4 * dt,
       'tau_ro' : 1*dt,
       'tau_star' : dt,

    	'N' : 500, 'T' : 800, 'I' : 4, 'O' : 3,

	'dv' : 0.05 , 'Vo' : -4, 'h' : -8, 's_inh' : 100,

       'gamma' : .99,
       'lerp'  : 0.01,

       'sigma_Jrec' : 0.0,
       'sigma_Jout' : 0.001,

       'alpha' : 0.1*10*.5*.5*.5,
       'alpha_rout' : 0.001,#0.01*.5*.5*.5,

	   'sigma_input' : 10.,
       'sigma_teach' : 10.,
       'hidden_steps' : 1,

       'policy_thr_tau' : 1,

       'outsig' : True,
       'step_mode' : 'amax',

       'epochs'     : 0,
       'epochs_out' : 0,

       'clump'    : False,
       'validate' : False,
       'feedback' : 'diagonal',
       'verbose'  : True,
       'rank'     : None
}


PONG_V4_PAR_SKIP4 = {'dt' : dt,
       'tau_m' : 6 * dt,
       'tau_s' : 4 * dt,
       'tau_ro' : 1 * dt,
       'tau_star' : dt,

    	'N' : 500, 'T' : 800, 'I' : 84*84, 'O' : 3,

	'dv' : 0.05 , 'Vo' : -4, 'h' : -8, 's_inh' : 100,

       'gamma' : .99,
       'lerp'  : 0.01,

       'sigma_Jrec' : 0.0,
       'sigma_Jout' : 0.001,

       'alpha' : 0.1*10,
       'alpha_rout' : 0.01,

	   'sigma_input' : .5,
       'sigma_teach' : 10.,
       'hidden_steps' : 1,

       'policy_thr_tau' : 1,

       'outsig' : True,
       'step_mode' : 'amax',

       'epochs'     : 0,
       'epochs_out' : 0,

       'clump'    : False,
       'validate' : False,
       'feedback' : 'diagonal',
       'verbose'  : True,
       'rank'     : None
}


PONG_V4_PAR_ = {'dt' : dt,
       'tau_m' : 6 * dt,
       'tau_s' : 4 * dt,
       'tau_ro' : 1*dt,
       'tau_star' : dt,

    	'N' : 500, 'T' : 800, 'I' : 128, 'O' : 3,

	'dv' : 0.05 , 'Vo' : -4, 'h' : -8, 's_inh' : 100,

       'gamma' : .99,
       'lerp'  : 0.01,

       'sigma_Jrec' : 0.0,
       'sigma_Jout' : 0.001,

       'alpha' : 0.1*10*.5*.5*.5,
       'alpha_rout' : 0.01*.5*.5*.5,

	   'sigma_input' : 2.,
       'sigma_teach' : 10.,
       'hidden_steps' : 1,

       'policy_thr_tau' : 1,

       'outsig' : True,
       'step_mode' : 'amax',

       'epochs'     : 0,
       'epochs_out' : 0,

       'clump'    : False,
       'validate' : False,
       'feedback' : 'diagonal',
       'verbose'  : True,
       'rank'     : None
}

CARTPOLE_V0_PAR = {'dt' : dt,
       'tau_m' : 6. * dt,
       'tau_s' : 2 * dt,
       'tau_ro' : 1.*dt,
       'tau_star' : 5.*dt,

    	'N' : 400, 'T' : 800, 'I' : 4, 'O' : 2,

	'dv' : 1. , 'Vo' : -4, 'h' : -8, 's_inh' : 100,

       'gamma' : .99,
       'lerp'  : 0.01,

       'sigma_Jrec' : 0.0,
       'sigma_Jout' : 0.001,

       'alpha' : 0.1*10*.5*.5*.5,
       'alpha_rout' : 0.01*.5*.5*.5,

	   'sigma_input' : 20.,#30.,
       'sigma_teach' : 10.,
       'hidden_steps' : 1,

       'policy_thr_tau' : 1,

       'outsig' : True,
       'step_mode' : 'amax',

       'epochs'     : 0,
       'epochs_out' : 0,

       'clump'    : False,
       'validate' : False,
       'feedback' : 'diagonal',
       'verbose'  : True,
       'rank'     : None
}

BUTTONFOOD_V0_AGENT = {
       'dt' : dt,
       'tau_m'    : 6 * dt,
       'tau_s'    : 4 * dt,
       'tau_ro'   : 1 * dt,
       'tau_star' : 1 * dt,

       'N' : 500, 'I' : 50, 'O' : 8,

       'dv' : 0.05 , 'Vo' : -4, 'h' : -8, 's_inh' : 100,

       'gamma' : .99,
       'lerp'  : 0.01,

       'sigma_Jrec' : 0.0,
       'sigma_Jout' : 0.001,

       'alpha_rec' : 0.000,
       'alpha_out' : 0.005,

       'sigma_input' : 10.,
       'sigma_teach' : 10.,
       'hidden_steps' : 1,

       'outsig' : True,
       'step_mode' : 'prob',
}


BUTTONFOOD_V0_PLANNER = {
       'dt' : dt,
       'tau_m'    : 6 * dt,
       'tau_s'    : 4 * dt,
       'tau_ro'   : 1 * dt,
       'tau_star' : 1 * dt,

       'N' : 500, 'I' : 50, 'O' : 8,

       'dv' : 0.05 , 'Vo' : -4, 'h' : -8, 's_inh' : 100,

       'gamma' : .99,
       'lerp'  : 0.01,

       'sigma_Jrec' : 0.0,
       'sigma_Jout' : 0.001,

       'alpha_rec' : 0.000,
       'alpha_out' : 1e-3,

       'sigma_input' : 10.,
       'sigma_teach' : 10.,
       'hidden_steps' : 1,

       'outsig' : True,
       'step_mode' : 'prob',
}

Config = {
    'Pong-ramDeterministic-v0' : PONG_V4_PAR_I4,
    'ButtonFood-v0_Agent'   : BUTTONFOOD_V0_AGENT,
    'ButtonFood-v0_Planner' : BUTTONFOOD_V0_PLANNER,
}