#include "message.h"
#include "kenneth.h"

#include "common/factorization/utils/divisors.h"
#include "common/factorization/utils/factorize.h"
#include "common/modular/static/modular.h"
#include "common/stl/base.h"

using TModular = Modular<>;

int main_kenneth()
{
	int nodes = NumberOfNodes();
	int node_id = MyNodeId();

	vector<int64_t> vs(nodes + 1);
	vs[0] = 0;
	for (int i = 0; i < nodes; ++i)
	{
		vs[i + 1] = vs[i] + GetPieceLength(i);
	}
	int64_t first = vs[node_id], last = vs[node_id + 1], ln = last - first, l = vs[nodes];

	vector<char> vstr(ln);
	for (int64_t i = first; i < last; ++i)
	{
		vstr[i - first] = GetSignalCharacter(i);
	}
	vector<TModular> vh(ln + 1);
	TModular one(1), mm = TModular(0x0e3779b9);
	TModular mb = mm.PowU(first), mc = mb;
	vh[0] = 0;
	for (int64_t i = 0; i < ln; ++i)
	{
		vh[i + 1] = vh[i] + mc * TModular(vstr[i]);
		mc *= mm;
	}
    TModular mmsum = mm.PowU(l) - one;
    assert(mmsum.Get() != 0);

    if (node_id != 0)
    {
        Receive(node_id - 1);
        vh[0].SetS(GetInt(node_id - 1));
    }
    PutInt((node_id + 1) % nodes, int((vh[0] + vh[ln]).Get()));
    Send((node_id + 1) % nodes);
    for (int64_t i = 1; i <= ln; ++i)
    {
        vh[i] += vh[0];
    }

    TModular sum;
    if (node_id == 0)
    {
        Receive(nodes - 1);
        sum.SetS(GetInt(nodes - 1));
        PutInt(1, int(sum.Get()));
        Send(1);
    }
    else
    {
        Receive(node_id - 1);
        sum.SetS(GetInt(node_id - 1));
        if (node_id != nodes - 1)
        {
            PutInt(node_id + 1, int(sum.Get()));
            Send(node_id + 1);
        }
    }

    auto vd = GetDivisors(Factorize(l));
    sort(vd.begin(), vd.end());
    vector<char> vok(vd.size(), 1);
    for (unsigned k = 0; k < vd.size(); ++k)
    {
        int64_t d = vd[k];
        TModular mmpowd = mm.PowU(d);
        TModular mmsumd = (mmpowd - one);
        TModular target = sum * mmsumd / mmsum;
        int64_t j = (first / d) + 1;
        TModular mmsumjd = (mm.PowU(j * d) - one);
        for (; j * d <= last; ++j)
        {            
            if (vh[j * d - first] * mmsumd != target * mmsumjd)
            {
                vok[k] = 0;
                break;
            }
            mmsumjd = mmsumjd * mmpowd + mmsumd;
            // break; // We really don't need to check all, should be fine with high probability.
        }
    }

    if (node_id != 0)
    {
        Receive(node_id - 1);
        for (unsigned k = 0; k < vd.size(); ++k)
            vok[k] *= GetChar(node_id - 1);
    }
    for (unsigned k = 0; k < vd.size(); ++k)
        PutChar((node_id + 1) % nodes, vok[k]);
    Send((node_id + 1) % nodes);
    
    if (node_id == 0)
    {
        int64_t d = 0;
        Receive(nodes - 1);
        for (unsigned k = 0; k < vd.size(); ++k)
        {
            vok[k] = GetChar(nodes - 1);
            if (vok[k] && (d == 0))
                d = vd[k];
        }
        assert(d > 0);
        cout << d << endl;
    }

    return 0;
}
