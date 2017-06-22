* 每一列分别代表什么意思？"SettlePrice" 一直等于-1是什么意思？
* "InstrumentID"中四位数字代表交割年月，期货合约交割日期为当月月中，合约上市交易日期为交割日期前一年。
* 铜：cu，铝：al，锌：zn，铅：pb，镍：ni，锡：sn，黄金：au，白银：ag，螺纹钢：rb，线材：wr，热轧卷板：hc， 燃料油：fu，沥青：bu，天然橡胶：ru
* 数据来自于上海交易所，[交易时间](http://www.guzhiwang.com/html/201107/11/20110711141403.htm)，周一至周五，上午：第一节9:00-10:15；第二节 10:30-11:30 下午：第一节 1:30-3:00。[晚上](http://www.xuexila.com/licai/qihuo/1377285.html)(1)夜盘交易时间：21:00-02:30;夜盘交易品种：黄金、白银。(2) 夜盘交易时间：21:00-01:00;夜盘交易品种：铜、铝、锌、铅。 数据里的UpdateTime是北京时间.文件里是交易日期从2016/01到2017/12的期货数据。数据更新频率一般为一秒钟二到三次。

* 怎样把不同的future放在同一个group里研究，应该不能直接把所有同一种商品的futures都当成是同一个prcie series吧？


* 有数据早于9：00，晚于15:00，怎么处理？
* 是不是应该把白天的data分成三个部分来计算variation，然后加在一起算vol？annualize return, 乘以sqrt（24/3.75*252）? 
* 应该用哪个price来计算volatility，用LastPrice吗
* 我们要研究哪一个频率的vol，是daily吗？
* 用Zhou Bin文章里的方法计算vol，需要选取一个lag，是应该选使得vol达到local minimum的lag吗？

* Concate column "Date" and "UpdateTime", equals to "DateTime"
* "timestamp" in data is in milisecond, convert to second then to datetime 
* beautify datetime in plot
	plt.gcf().autofmt_xdate()
 

